"""
策略回放器测试
Strategy Replayer Tests

测试:
- 信号加载与回放
- 交易执行逻辑
- 持仓管理
- 绩效计算
- 多种回放模式

作者: TraderX Team
创建: 2025-12-24
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.strategy_replayer import (
    StrategyReplayer,
    ReplayConfig,
    ReplayMode,
    ReplaySpeed,
    ReplaySignal,
    ReplayTrade,
    ReplayPosition,
    ReplayResult,
    create_strategy_replayer,
    replay_signals,
)


# ============ Fixtures ============

@pytest.fixture
def sample_signals():
    """样本信号数据"""
    base_time = datetime(2024, 1, 1, 9, 30)
    signals = []

    for i in range(10):
        signals.append({
            'timestamp': base_time + timedelta(hours=i),
            'symbol': 'AAPL' if i % 2 == 0 else 'MSFT',
            'direction': 'long' if i % 3 != 0 else 'short',
            'strength': 0.6 + (i % 5) * 0.05,
            'confidence': 0.7 + (i % 4) * 0.05,
            'source': 'test_strategy',
            'price': 150.0 + i * 0.5,
        })

    return signals


@pytest.fixture
def sample_prices():
    """样本价格数据"""
    dates = pd.date_range('2024-01-01 09:30', periods=20, freq='h')
    prices = pd.DataFrame({
        'close': [150.0 + i * 0.5 for i in range(20)],
        'open': [149.5 + i * 0.5 for i in range(20)],
        'high': [151.0 + i * 0.5 for i in range(20)],
        'low': [149.0 + i * 0.5 for i in range(20)],
        'volume': [1000000] * 20,
    }, index=dates)
    return prices


@pytest.fixture
def default_config():
    """默认配置"""
    return ReplayConfig(
        mode=ReplayMode.BACKTEST,
        speed=ReplaySpeed.INSTANT,
        initial_capital=1_000_000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        min_signal_strength=0.3,
        min_signal_confidence=0.5,
        max_position_weight=0.20,
        max_daily_trades=50,
    )


# ============ ReplayConfig Tests ============

class TestReplayConfig:
    """回放配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ReplayConfig()

        assert config.mode == ReplayMode.BACKTEST
        assert config.speed == ReplaySpeed.INSTANT
        assert config.initial_capital == 1_000_000
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005

    def test_custom_config(self):
        """测试自定义配置"""
        config = ReplayConfig(
            mode=ReplayMode.PAPER,
            initial_capital=500_000,
            max_position_weight=0.10,
        )

        assert config.mode == ReplayMode.PAPER
        assert config.initial_capital == 500_000
        assert config.max_position_weight == 0.10


# ============ ReplaySignal Tests ============

class TestReplaySignal:
    """回放信号测试"""

    def test_signal_creation(self):
        """测试信号创建"""
        signal = ReplaySignal(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol='AAPL',
            direction='long',
            strength=0.8,
            confidence=0.7,
            source='test',
        )

        assert signal.symbol == 'AAPL'
        assert signal.direction == 'long'
        assert signal.strength == 0.8

    def test_signal_to_dict(self):
        """测试信号序列化"""
        signal = ReplaySignal(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol='MSFT',
            direction='short',
            strength=0.6,
            confidence=0.75,
            source='test',
            price_at_signal=350.0,
        )

        d = signal.to_dict()
        assert d['symbol'] == 'MSFT'
        assert d['direction'] == 'short'
        assert d['price_at_signal'] == 350.0


# ============ StrategyReplayer Tests ============

class TestStrategyReplayer:
    """策略回放器测试"""

    def test_create_replayer(self, default_config):
        """测试创建回放器"""
        replayer = StrategyReplayer(default_config)

        assert replayer.config.mode == ReplayMode.BACKTEST
        assert replayer._capital == 1_000_000
        assert len(replayer._signals) == 0

    def test_load_signals(self, default_config, sample_signals):
        """测试加载信号"""
        replayer = StrategyReplayer(default_config)
        count = replayer.load_signals(sample_signals)

        assert count == 10
        assert len(replayer._signals) == 10
        # 验证按时间排序
        for i in range(1, len(replayer._signals)):
            assert replayer._signals[i].timestamp >= replayer._signals[i-1].timestamp

    def test_load_prices(self, default_config, sample_prices):
        """测试加载价格"""
        replayer = StrategyReplayer(default_config)
        count = replayer.load_prices(sample_prices)

        assert count == 20
        assert replayer._prices is not None


class TestSignalValidation:
    """信号验证测试"""

    def test_validate_signal_pass(self, default_config):
        """测试信号验证通过"""
        replayer = StrategyReplayer(default_config)

        signal = ReplaySignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='long',
            strength=0.6,
            confidence=0.7,
            source='test',
        )

        assert replayer._validate_signal(signal) is True

    def test_validate_signal_low_strength(self, default_config):
        """测试低强度信号被拒绝"""
        replayer = StrategyReplayer(default_config)

        signal = ReplaySignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='long',
            strength=0.2,  # 低于阈值0.3
            confidence=0.7,
            source='test',
        )

        assert replayer._validate_signal(signal) is False

    def test_validate_signal_low_confidence(self, default_config):
        """测试低置信度信号被拒绝"""
        replayer = StrategyReplayer(default_config)

        signal = ReplaySignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='long',
            strength=0.6,
            confidence=0.3,  # 低于阈值0.5
            source='test',
        )

        assert replayer._validate_signal(signal) is False

    def test_validate_signal_neutral_rejected(self, default_config):
        """测试中性信号被拒绝"""
        replayer = StrategyReplayer(default_config)

        signal = ReplaySignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='neutral',
            strength=0.8,
            confidence=0.9,
            source='test',
        )

        assert replayer._validate_signal(signal) is False


class TestReplayExecution:
    """回放执行测试"""

    @pytest.mark.asyncio
    async def test_run_empty_signals(self, default_config):
        """测试空信号回放"""
        replayer = StrategyReplayer(default_config)
        result = await replayer.run()

        assert result.total_trades == 0
        assert result.final_capital == result.initial_capital

    @pytest.mark.asyncio
    async def test_run_single_signal(self, default_config):
        """测试单个信号回放"""
        replayer = StrategyReplayer(default_config)

        signals = [{
            'timestamp': datetime(2024, 1, 1, 10, 0),
            'symbol': 'AAPL',
            'direction': 'long',
            'strength': 0.8,
            'confidence': 0.7,
            'source': 'test',
            'price': 150.0,
        }]

        replayer.load_signals(signals)
        result = await replayer.run()

        assert result.signals_processed == 1
        assert result.signals_executed == 1
        assert len(result.trades) >= 1

    @pytest.mark.asyncio
    async def test_run_multiple_signals(self, default_config, sample_signals):
        """测试多信号回放"""
        replayer = StrategyReplayer(default_config)
        replayer.load_signals(sample_signals)

        result = await replayer.run()

        assert result.signals_processed == 10
        assert result.signals_executed > 0
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_daily_trade_limit(self):
        """测试日交易限制"""
        config = ReplayConfig(
            max_daily_trades=2,
            min_signal_strength=0.0,
            min_signal_confidence=0.0,
        )
        replayer = StrategyReplayer(config)

        # 创建同一天的多个信号
        base_time = datetime(2024, 1, 1, 10, 0)
        signals = []
        for i in range(5):
            signals.append({
                'timestamp': base_time + timedelta(hours=i),
                'symbol': f'SYM{i}',
                'direction': 'long',
                'strength': 0.8,
                'confidence': 0.8,
                'source': 'test',
                'price': 100.0,
            })

        replayer.load_signals(signals)
        result = await replayer.run()

        assert result.signals_executed <= 2
        assert result.signals_rejected >= 3


class TestPositionManagement:
    """持仓管理测试"""

    @pytest.mark.asyncio
    async def test_open_position(self, default_config):
        """测试开仓"""
        replayer = StrategyReplayer(default_config)

        signal = ReplaySignal(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol='AAPL',
            direction='long',
            strength=0.8,
            confidence=0.7,
            source='test',
            price_at_signal=150.0,
        )

        replayer._current_prices['AAPL'] = 150.0
        trade = await replayer._execute_signal(signal)

        assert trade is not None
        assert trade.symbol == 'AAPL'
        assert trade.side == 'buy'
        assert 'AAPL' in replayer._positions

    @pytest.mark.asyncio
    async def test_close_position(self, default_config):
        """测试平仓"""
        replayer = StrategyReplayer(default_config)

        # 先开仓
        replayer._positions['AAPL'] = ReplayPosition(
            symbol='AAPL',
            side='long',
            quantity=100,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 1, 10, 0),
            current_price=155.0,
        )

        replayer._current_prices['AAPL'] = 155.0
        trade = await replayer._close_position(
            'AAPL',
            datetime(2024, 1, 1, 14, 0),
            155.0,
        )

        assert trade is not None
        assert trade.side == 'sell'
        assert trade.pnl > 0  # 盈利
        assert 'AAPL' not in replayer._positions

    @pytest.mark.asyncio
    async def test_close_all_positions(self, default_config):
        """测试平掉所有持仓"""
        replayer = StrategyReplayer(default_config)

        # 创建多个持仓
        replayer._positions['AAPL'] = ReplayPosition(
            symbol='AAPL', side='long', quantity=100,
            entry_price=150.0, entry_time=datetime.now(),
        )
        replayer._positions['MSFT'] = ReplayPosition(
            symbol='MSFT', side='short', quantity=50,
            entry_price=350.0, entry_time=datetime.now(),
        )

        replayer._current_prices = {'AAPL': 155.0, 'MSFT': 345.0}
        replayer._current_timestamp = datetime.now()

        await replayer._close_all_positions()

        assert len(replayer._positions) == 0


class TestPerformanceCalculation:
    """绩效计算测试"""

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, default_config):
        """测试胜率计算"""
        replayer = StrategyReplayer(default_config)

        # 模拟一些交易
        replayer._trades = [
            ReplayTrade(
                trade_id='1', timestamp=datetime.now(), symbol='AAPL',
                side='buy', quantity=100, price=150.0, commission=0.1,
                slippage=0.01, pnl=500.0, pnl_pct=0.03,
                exit_timestamp=datetime.now(), exit_price=155.0,
            ),
            ReplayTrade(
                trade_id='2', timestamp=datetime.now(), symbol='MSFT',
                side='buy', quantity=50, price=350.0, commission=0.1,
                slippage=0.01, pnl=-200.0, pnl_pct=-0.01,
                exit_timestamp=datetime.now(), exit_price=345.0,
            ),
        ]

        result = replayer._calculate_result(
            datetime.now() - timedelta(hours=1),
            datetime.now(),
            3600.0,
        )

        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == 0.5

    @pytest.mark.asyncio
    async def test_profit_factor_calculation(self, default_config):
        """测试盈亏比计算"""
        replayer = StrategyReplayer(default_config)

        replayer._trades = [
            ReplayTrade(
                trade_id='1', timestamp=datetime.now(), symbol='AAPL',
                side='buy', quantity=100, price=150.0, commission=0.1,
                slippage=0.01, pnl=1000.0, pnl_pct=0.06,
                exit_timestamp=datetime.now(), exit_price=160.0,
            ),
            ReplayTrade(
                trade_id='2', timestamp=datetime.now(), symbol='MSFT',
                side='buy', quantity=50, price=350.0, commission=0.1,
                slippage=0.01, pnl=-250.0, pnl_pct=-0.01,
                exit_timestamp=datetime.now(), exit_price=345.0,
            ),
        ]

        result = replayer._calculate_result(
            datetime.now() - timedelta(hours=1),
            datetime.now(),
            3600.0,
        )

        # profit_factor = 1000 / 250 = 4.0
        assert result.profit_factor == 4.0


class TestReplayModes:
    """回放模式测试"""

    def test_backtest_mode(self):
        """测试回测模式"""
        config = ReplayConfig(mode=ReplayMode.BACKTEST)
        replayer = StrategyReplayer(config)

        assert replayer.config.mode == ReplayMode.BACKTEST

    def test_paper_mode(self):
        """测试模拟交易模式"""
        config = ReplayConfig(mode=ReplayMode.PAPER)
        replayer = StrategyReplayer(config)

        assert replayer.config.mode == ReplayMode.PAPER

    def test_validation_mode(self):
        """测试验证模式"""
        config = ReplayConfig(mode=ReplayMode.VALIDATION)
        replayer = StrategyReplayer(config)

        assert replayer.config.mode == ReplayMode.VALIDATION


class TestCallbacks:
    """回调测试"""

    @pytest.mark.asyncio
    async def test_on_signal_callback(self, default_config):
        """测试信号回调"""
        replayer = StrategyReplayer(default_config)

        callback_signals = []
        replayer.set_callbacks(on_signal=lambda s: callback_signals.append(s))

        signals = [{
            'timestamp': datetime(2024, 1, 1, 10, 0),
            'symbol': 'AAPL',
            'direction': 'long',
            'strength': 0.8,
            'confidence': 0.7,
            'source': 'test',
            'price': 150.0,
        }]

        replayer.load_signals(signals)
        await replayer.run()

        assert len(callback_signals) == 1
        assert callback_signals[0].symbol == 'AAPL'

    @pytest.mark.asyncio
    async def test_on_trade_callback(self, default_config):
        """测试交易回调"""
        replayer = StrategyReplayer(default_config)

        callback_trades = []
        replayer.set_callbacks(on_trade=lambda t: callback_trades.append(t))

        signals = [{
            'timestamp': datetime(2024, 1, 1, 10, 0),
            'symbol': 'AAPL',
            'direction': 'long',
            'strength': 0.8,
            'confidence': 0.7,
            'source': 'test',
            'price': 150.0,
        }]

        replayer.load_signals(signals)
        await replayer.run()

        assert len(callback_trades) >= 1


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_strategy_replayer_default(self):
        """测试默认创建"""
        replayer = create_strategy_replayer()

        assert replayer.config.mode == ReplayMode.BACKTEST
        assert replayer.config.initial_capital == 1_000_000

    def test_create_strategy_replayer_custom(self):
        """测试自定义创建"""
        replayer = create_strategy_replayer(
            mode='paper',
            initial_capital=500_000,
            max_position_weight=0.10,
        )

        assert replayer.config.mode == ReplayMode.PAPER
        assert replayer.config.initial_capital == 500_000
        assert replayer.config.max_position_weight == 0.10

    @pytest.mark.asyncio
    async def test_replay_signals_convenience(self):
        """测试便捷回放函数"""
        signals = [
            {
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'symbol': 'AAPL',
                'direction': 'long',
                'strength': 0.8,
                'confidence': 0.7,
                'source': 'test',
                'price': 150.0,
            },
            {
                'timestamp': datetime(2024, 1, 1, 11, 0),
                'symbol': 'MSFT',
                'direction': 'short',
                'strength': 0.7,
                'confidence': 0.6,
                'source': 'test',
                'price': 350.0,
            },
        ]

        result = await replay_signals(signals)

        assert isinstance(result, ReplayResult)
        assert result.signals_processed == 2


class TestStatistics:
    """统计功能测试"""

    def test_get_statistics(self, default_config):
        """测试获取统计"""
        replayer = StrategyReplayer(default_config)

        stats = replayer.get_statistics()

        assert 'is_running' in stats
        assert 'capital' in stats
        assert 'equity' in stats
        assert stats['is_running'] is False

    def test_get_equity(self, default_config):
        """测试获取权益"""
        replayer = StrategyReplayer(default_config)

        # 无持仓时权益等于资金
        assert replayer.get_equity() == 1_000_000

        # 添加持仓
        replayer._positions['AAPL'] = ReplayPosition(
            symbol='AAPL', side='long', quantity=100,
            entry_price=150.0, entry_time=datetime.now(),
            current_price=155.0,
        )

        equity = replayer.get_equity()
        assert equity > 1_000_000  # 权益应增加

    def test_get_trades(self, default_config):
        """测试获取交易记录"""
        replayer = StrategyReplayer(default_config)

        assert len(replayer.get_trades()) == 0

        replayer._trades.append(ReplayTrade(
            trade_id='1', timestamp=datetime.now(), symbol='AAPL',
            side='buy', quantity=100, price=150.0, commission=0.1,
            slippage=0.01,
        ))

        trades = replayer.get_trades()
        assert len(trades) == 1
        assert trades[0].symbol == 'AAPL'


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_signals(self, default_config):
        """测试空信号列表"""
        replayer = StrategyReplayer(default_config)
        replayer.load_signals([])

        result = await replayer.run()

        assert result.total_trades == 0
        assert result.signals_processed == 0

    @pytest.mark.asyncio
    async def test_stop_during_replay(self, default_config):
        """测试回放中停止"""
        replayer = StrategyReplayer(default_config)

        signals = []
        for i in range(100):
            signals.append({
                'timestamp': datetime(2024, 1, 1) + timedelta(hours=i),
                'symbol': 'AAPL',
                'direction': 'long',
                'strength': 0.8,
                'confidence': 0.7,
                'source': 'test',
                'price': 150.0 + i,
            })

        replayer.load_signals(signals)

        # 启动回放任务
        import asyncio
        async def stop_after_delay():
            await asyncio.sleep(0.01)
            replayer.stop()

        asyncio.create_task(stop_after_delay())
        result = await replayer.run()

        # 应该在全部处理完之前停止
        assert result.signals_processed < 100 or replayer._is_running is False

    def test_result_to_dict(self, default_config):
        """测试结果序列化"""
        result = ReplayResult(
            mode=ReplayMode.BACKTEST,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            duration_seconds=86400,
            initial_capital=1_000_000,
            final_capital=1_050_000,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=50_000,
            win_rate=0.6,
        )

        d = result.to_dict()

        assert d['mode'] == 'backtest'
        assert d['total_trades'] == 10
        assert d['win_rate'] == 0.6


# ============ 运行测试 ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
