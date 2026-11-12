"""
策略信号执行器单元测试

测试:
- 信号创建和验证
- 订单生成
- 仓位计算
- 模拟执行
- 批量执行
- 状态管理
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

# 导入被测模块
import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.signal_executor import (
    SignalExecutor,
    TradingSignal,
    TradeOrder,
    ExecutionResult,
    ExecutorConfig,
    SignalDirection,
    OrderSide,
    OrderType,
    OrderStatus,
    ExecutionMode,
    EqualWeightSizer,
    VolatilityAdjustedSizer,
    create_signal_executor,
    create_trading_signal,
)


# ============ Fixtures ============

@pytest.fixture
def executor_config():
    """基础执行器配置"""
    return ExecutorConfig(
        java_api_base_url="http://localhost:8080",
        auth_token="test-token",
        execution_mode=ExecutionMode.PAPER,
        min_signal_strength=0.3,
        min_signal_confidence=0.5,
        min_trade_value=100.0,
        max_position_weight=0.20,
        max_daily_trades=50,
    )


@pytest.fixture
def signal_only_config():
    """仅信号模式配置"""
    return ExecutorConfig(
        execution_mode=ExecutionMode.SIGNAL_ONLY,
        min_signal_strength=0.3,
        min_signal_confidence=0.5,
    )


@pytest.fixture
def basic_signal():
    """基础交易信号"""
    return TradingSignal(
        signal_id="SIG-TEST001",
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strength=0.8,
        confidence=0.7,
        source="test_strategy",
        generated_at=datetime.now(),
    )


@pytest.fixture
def strong_buy_signal():
    """强买入信号"""
    return TradingSignal(
        signal_id="SIG-STRONG001",
        symbol="MSFT",
        direction=SignalDirection.STRONG_LONG,
        strength=0.95,
        confidence=0.9,
        source="momentum_strategy",
        generated_at=datetime.now(),
        target_price=400.0,
        stop_loss=380.0,
        take_profit=420.0,
    )


@pytest.fixture
def weak_signal():
    """弱信号(不应触发交易)"""
    return TradingSignal(
        signal_id="SIG-WEAK001",
        symbol="WEAK",
        direction=SignalDirection.LONG,
        strength=0.2,  # 低于阈值
        confidence=0.4,  # 低于阈值
        source="test",
        generated_at=datetime.now(),
    )


@pytest.fixture
def neutral_signal():
    """中性信号"""
    return TradingSignal(
        signal_id="SIG-NEUTRAL001",
        symbol="NEUTRAL",
        direction=SignalDirection.NEUTRAL,
        strength=0.5,
        confidence=0.5,
        source="test",
        generated_at=datetime.now(),
    )


@pytest.fixture
def expired_signal():
    """过期信号"""
    return TradingSignal(
        signal_id="SIG-OLD001",
        symbol="OLD",
        direction=SignalDirection.LONG,
        strength=0.8,
        confidence=0.8,
        source="test",
        generated_at=datetime.now() - timedelta(hours=48),  # 超过24小时
    )


@pytest.fixture
def multiple_signals():
    """多个信号"""
    return [
        TradingSignal(
            signal_id=f"SIG-MULTI{i}",
            symbol=symbol,
            direction=SignalDirection.LONG if i % 2 == 0 else SignalDirection.SHORT,
            strength=0.7 + i * 0.05,
            confidence=0.6 + i * 0.05,
            source="multi_test",
            generated_at=datetime.now(),
        )
        for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN"])
    ]


# ============ 信号创建测试 ============

class TestTradingSignal:
    """交易信号测试"""

    def test_basic_creation(self, basic_signal):
        """测试基础信号创建"""
        assert basic_signal.signal_id == "SIG-TEST001"
        assert basic_signal.symbol == "AAPL"
        assert basic_signal.direction == SignalDirection.LONG
        assert basic_signal.strength == 0.8
        assert basic_signal.confidence == 0.7

    def test_normalized_value_long(self, basic_signal):
        """测试看多信号标准化值"""
        # LONG = 0.5 * strength * confidence
        expected = 0.5 * 0.8 * 0.7
        assert abs(basic_signal.normalized_value - expected) < 0.001

    def test_normalized_value_strong_long(self, strong_buy_signal):
        """测试强看多信号标准化值"""
        # STRONG_LONG = 1.0 * strength * confidence
        expected = 1.0 * 0.95 * 0.9
        assert abs(strong_buy_signal.normalized_value - expected) < 0.001

    def test_normalized_value_short(self):
        """测试看空信号标准化值"""
        signal = TradingSignal(
            signal_id="SHORT001",
            symbol="TEST",
            direction=SignalDirection.SHORT,
            strength=0.8,
            confidence=0.8,
            source="test",
            generated_at=datetime.now(),
        )
        # SHORT = -0.5 * strength * confidence
        expected = -0.5 * 0.8 * 0.8
        assert abs(signal.normalized_value - expected) < 0.001

    def test_normalized_value_neutral(self, neutral_signal):
        """测试中性信号标准化值"""
        assert neutral_signal.normalized_value == 0.0


class TestCreateTradingSignal:
    """工厂函数测试"""

    def test_create_basic_signal(self):
        """测试创建基础信号"""
        signal = create_trading_signal(
            symbol="AAPL",
            direction="long",
            strength=0.8,
            confidence=0.7,
            source="test",
        )

        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.strength == 0.8
        assert signal.signal_id.startswith("SIG-")

    def test_create_signal_with_bounds(self):
        """测试信号强度边界处理"""
        # 超出范围应被限制
        signal = create_trading_signal(
            symbol="TEST",
            direction="long",
            strength=1.5,  # 超出
            confidence=-0.1,  # 超出
            source="test",
        )

        assert signal.strength == 1.0  # 被限制为1
        assert signal.confidence == 0.0  # 被限制为0

    def test_create_all_directions(self):
        """测试所有方向类型"""
        directions = ["strong_long", "long", "neutral", "short", "strong_short"]
        expected = [
            SignalDirection.STRONG_LONG,
            SignalDirection.LONG,
            SignalDirection.NEUTRAL,
            SignalDirection.SHORT,
            SignalDirection.STRONG_SHORT,
        ]

        for dir_str, expected_dir in zip(directions, expected):
            signal = create_trading_signal(
                symbol="TEST",
                direction=dir_str,
                strength=0.5,
                confidence=0.5,
                source="test",
            )
            assert signal.direction == expected_dir


# ============ 订单测试 ============

class TestTradeOrder:
    """交易订单测试"""

    def test_order_creation(self):
        """测试订单创建"""
        order = TradeOrder(
            order_id="ORD-001",
            signal_id="SIG-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.order_id == "ORD-001"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0

    def test_order_to_dict(self):
        """测试订单转字典"""
        order = TradeOrder(
            order_id="ORD-001",
            signal_id="SIG-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        data = order.to_dict()

        assert data['order_id'] == "ORD-001"
        assert data['symbol'] == "AAPL"
        assert data['side'] == "buy"
        assert data['order_type'] == "limit"
        assert data['limit_price'] == 150.0


# ============ 仓位计算器测试 ============

class TestEqualWeightSizer:
    """等权重仓位计算器测试"""

    def test_basic_sizing(self, basic_signal, executor_config):
        """测试基础仓位计算"""
        sizer = EqualWeightSizer(base_weight=0.05)

        size = sizer.calculate_position_size(
            signal=basic_signal,
            portfolio_value=1000000,
            current_positions={},
            config=executor_config,
        )

        # 0.05 * 0.8 * 0.7 = 0.028
        expected = 1000000 * 0.05 * 0.8 * 0.7
        assert abs(size - expected) < 0.01

    def test_max_weight_limit(self, strong_buy_signal, executor_config):
        """测试最大权重限制"""
        sizer = EqualWeightSizer(base_weight=0.50)  # 很高的基础权重

        size = sizer.calculate_position_size(
            signal=strong_buy_signal,
            portfolio_value=1000000,
            current_positions={},
            config=executor_config,
        )

        # 应该被限制在max_position_weight (0.20)
        max_size = 1000000 * executor_config.max_position_weight
        assert size <= max_size


class TestVolatilityAdjustedSizer:
    """波动率调整仓位计算器测试"""

    def test_low_volatility_scaling(self, basic_signal, executor_config):
        """测试低波动率放大仓位"""
        sizer = VolatilityAdjustedSizer(target_vol=0.15)
        sizer.set_volatility("AAPL", 0.10)  # 低波动

        size = sizer.calculate_position_size(
            signal=basic_signal,
            portfolio_value=1000000,
            current_positions={},
            config=executor_config,
        )

        # 波动率调整系数 = 0.15 / 0.10 = 1.5
        # 应该比基础仓位大
        base_sizer = EqualWeightSizer()
        base_size = base_sizer.calculate_position_size(
            basic_signal, 1000000, {}, executor_config
        )

        assert size > base_size

    def test_high_volatility_scaling(self, basic_signal, executor_config):
        """测试高波动率缩小仓位"""
        sizer = VolatilityAdjustedSizer(target_vol=0.15)
        sizer.set_volatility("AAPL", 0.30)  # 高波动

        size = sizer.calculate_position_size(
            signal=basic_signal,
            portfolio_value=1000000,
            current_positions={},
            config=executor_config,
        )

        # 波动率调整系数 = 0.15 / 0.30 = 0.5
        base_sizer = EqualWeightSizer()
        base_size = base_sizer.calculate_position_size(
            basic_signal, 1000000, {}, executor_config
        )

        assert size < base_size


# ============ 执行器测试 ============

class TestSignalExecutor:
    """信号执行器测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self, executor_config):
        """测试启动和停止"""
        executor = SignalExecutor(config=executor_config)

        assert not executor.is_running

        await executor.start()
        assert executor.is_running

        await executor.stop()
        assert not executor.is_running

    @pytest.mark.asyncio
    async def test_signal_validation_pass(self, executor_config, basic_signal):
        """测试信号验证通过"""
        executor = SignalExecutor(config=executor_config)

        assert executor._validate_signal(basic_signal)

    @pytest.mark.asyncio
    async def test_signal_validation_fail_strength(self, executor_config, weak_signal):
        """测试信号强度不足被拒绝"""
        executor = SignalExecutor(config=executor_config)

        assert not executor._validate_signal(weak_signal)

    @pytest.mark.asyncio
    async def test_signal_validation_fail_neutral(self, executor_config, neutral_signal):
        """测试中性信号被拒绝"""
        executor = SignalExecutor(config=executor_config)

        assert not executor._validate_signal(neutral_signal)

    @pytest.mark.asyncio
    async def test_signal_validation_fail_expired(self, executor_config, expired_signal):
        """测试过期信号被拒绝"""
        executor = SignalExecutor(config=executor_config)

        assert not executor._validate_signal(expired_signal)

    @pytest.mark.asyncio
    async def test_submit_signal(self, executor_config, basic_signal):
        """测试提交信号"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor.submit_signal(basic_signal)
            assert result is True
            assert executor._signal_queue.qsize() == 1
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_submit_invalid_signal(self, executor_config, weak_signal):
        """测试提交无效信号被拒绝"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor.submit_signal(weak_signal)
            assert result is False
            assert executor._signal_queue.qsize() == 0
        finally:
            await executor.stop()


class TestPaperExecution:
    """模拟执行测试"""

    @pytest.mark.asyncio
    async def test_paper_execute_buy(self, executor_config, basic_signal):
        """测试模拟买入"""
        callback_called = []

        async def on_filled(result):
            callback_called.append(result)

        executor = SignalExecutor(
            config=executor_config,
            on_order_filled=on_filled,
        )
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=150.0,
            )

            assert result is not None
            assert result.success is True
            assert result.order.status == OrderStatus.FILLED
            assert result.order.side == OrderSide.BUY
            assert len(callback_called) == 1
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_paper_execute_sell(self, executor_config):
        """测试模拟卖出"""
        short_signal = TradingSignal(
            signal_id="SHORT001",
            symbol="TEST",
            direction=SignalDirection.SHORT,
            strength=0.8,
            confidence=0.8,
            source="test",
            generated_at=datetime.now(),
        )

        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=short_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            assert result is not None
            assert result.success is True
            assert result.order.side == OrderSide.SELL
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_paper_execute_commission(self, executor_config, basic_signal):
        """测试模拟交易手续费计算"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            assert result is not None
            # 手续费应该是 quantity * price * 0.001
            expected_commission = result.order.filled_quantity * 100.0 * 0.001
            assert abs(result.order.commission - expected_commission) < 0.01
        finally:
            await executor.stop()


class TestSignalOnlyMode:
    """仅信号模式测试"""

    @pytest.mark.asyncio
    async def test_signal_only_no_execution(self, signal_only_config, basic_signal):
        """测试仅信号模式不执行交易"""
        executor = SignalExecutor(config=signal_only_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            assert result is not None
            assert result.success is True
            assert result.order.status == OrderStatus.PENDING
            assert "Signal only mode" in result.order.message
        finally:
            await executor.stop()


class TestBatchExecution:
    """批量执行测试"""

    @pytest.mark.asyncio
    async def test_batch_execute(self, executor_config, multiple_signals):
        """测试批量执行"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            prices = {
                "AAPL": 150.0,
                "MSFT": 350.0,
                "GOOGL": 140.0,
                "AMZN": 180.0,
            }

            results = await executor.execute_signals_batch(
                signals=multiple_signals,
                portfolio_id=1,
                portfolio_value=1000000,
                prices=prices,
            )

            # 所有信号都应该被处理(都满足阈值)
            assert len(results) == 4

            # 检查订单方向
            for i, result in enumerate(results):
                expected_side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
                assert result.order.side == expected_side
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_batch_filter_invalid(self, executor_config):
        """测试批量执行过滤无效信号"""
        mixed_signals = [
            TradingSignal(  # 有效
                signal_id="VALID",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.8,
                source="test",
                generated_at=datetime.now(),
            ),
            TradingSignal(  # 无效 - 强度太低
                signal_id="INVALID",
                symbol="TEST",
                direction=SignalDirection.LONG,
                strength=0.1,
                confidence=0.1,
                source="test",
                generated_at=datetime.now(),
            ),
        ]

        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            results = await executor.execute_signals_batch(
                signals=mixed_signals,
                portfolio_id=1,
                portfolio_value=1000000,
                prices={"AAPL": 150.0, "TEST": 100.0},
            )

            # 只有有效信号被执行
            assert len(results) == 1
            assert results[0].order.symbol == "AAPL"
        finally:
            await executor.stop()


class TestStatistics:
    """统计功能测试"""

    @pytest.mark.asyncio
    async def test_get_statistics(self, executor_config):
        """测试获取统计信息"""
        executor = SignalExecutor(config=executor_config)

        stats = executor.get_statistics()

        assert 'is_running' in stats
        assert 'execution_mode' in stats
        assert 'total_orders' in stats
        assert 'filled_orders' in stats
        assert 'daily_trade_count' in stats

    @pytest.mark.asyncio
    async def test_statistics_after_trades(self, executor_config, basic_signal):
        """测试交易后统计更新"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            stats = executor.get_statistics()

            assert stats['total_orders'] == 1
            assert stats['filled_orders'] == 1
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_order_history(self, executor_config, basic_signal):
        """测试订单历史"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            history = executor.get_order_history()

            assert len(history) == 1
            assert history[0].symbol == "AAPL"
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_order_history_filter(self, executor_config, multiple_signals):
        """测试订单历史过滤"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            for signal in multiple_signals:
                await executor._execute_signal(
                    signal=signal,
                    portfolio_id=1,
                    portfolio_value=1000000,
                    current_price=100.0,
                )

            # 按标的过滤
            aapl_history = executor.get_order_history(symbol="AAPL")
            assert len(aapl_history) == 1
            assert aapl_history[0].symbol == "AAPL"
        finally:
            await executor.stop()


class TestDailyLimits:
    """日限制测试"""

    @pytest.mark.asyncio
    async def test_daily_trade_count(self, executor_config, basic_signal):
        """测试日交易计数"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            # 执行一笔交易
            await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=100.0,
            )

            # 日计数应该增加
            assert executor._daily_trade_count == 0  # 注意：_execute_signal 不更新日计数
            # 日计数在 _process_signals 中更新
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_daily_reset(self, executor_config):
        """测试日重置"""
        executor = SignalExecutor(config=executor_config)
        executor._daily_trade_count = 10
        executor._last_reset_date = (datetime.now() - timedelta(days=1)).date()

        executor._check_daily_reset()

        assert executor._daily_trade_count == 0
        assert executor._last_reset_date == datetime.now().date()


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_paper_executor(self):
        """测试创建模拟执行器"""
        executor = create_signal_executor(
            java_api_url="http://test:8080",
            execution_mode="paper",
        )

        assert executor.config.execution_mode == ExecutionMode.PAPER
        assert executor.config.java_api_base_url == "http://test:8080"

    def test_create_live_executor(self):
        """测试创建实盘执行器"""
        executor = create_signal_executor(
            java_api_url="http://prod:8080",
            auth_token="secret",
            execution_mode="live",
        )

        assert executor.config.execution_mode == ExecutionMode.LIVE
        assert executor.config.auth_token == "secret"

    def test_create_with_volatility_sizer(self):
        """测试创建波动率仓位执行器"""
        executor = create_signal_executor(
            execution_mode="paper",
            position_sizing="volatility",
        )

        assert isinstance(executor.position_sizer, VolatilityAdjustedSizer)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_min_trade_value_filter(self, executor_config):
        """测试最小交易金额过滤"""
        tiny_signal = TradingSignal(
            signal_id="TINY",
            symbol="TEST",
            direction=SignalDirection.LONG,
            strength=0.31,  # 刚过阈值
            confidence=0.51,  # 刚过阈值
            source="test",
            generated_at=datetime.now(),
        )

        # 配置很小的组合价值使得交易金额低于阈值
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=tiny_signal,
                portfolio_id=1,
                portfolio_value=100,  # 很小的组合价值
                current_price=100.0,
            )

            # 应该返回None因为交易金额太小
            assert result is None
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_no_price_execution(self, executor_config, basic_signal):
        """测试无价格执行"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=None,  # 无价格
            )

            assert result is not None
            # 应该使用默认价格
            assert result.order.filled_price == 100.0
        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_execution_result_to_dict(self, executor_config, basic_signal):
        """测试执行结果序列化"""
        executor = SignalExecutor(config=executor_config)
        await executor.start()

        try:
            result = await executor._execute_signal(
                signal=basic_signal,
                portfolio_id=1,
                portfolio_value=1000000,
                current_price=150.0,
            )

            data = result.to_dict()

            assert 'success' in data
            assert 'order' in data
            assert 'executed_at' in data
            assert data['success'] is True
        finally:
            await executor.stop()


# ============ 运行测试 ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
