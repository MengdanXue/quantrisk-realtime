"""
信号桥接器单元测试

测试:
- 信号过滤规则
- 信号转换
- 冷却机制
- 速率限制
- 批量处理
- 统计功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

# 导入被测模块
import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.signal_bridge import (
    SignalBridge,
    BridgeConfig,
    BridgeState,
    BridgedSignal,
    BridgeStatistics,
    SignalFilterResult,
    create_signal_bridge,
)
from quantrisk_realtime.realtime.signal_executor import (
    SignalExecutor,
    ExecutorConfig,
    ExecutionMode,
    TradingSignal,
    SignalDirection,
)


# ============ Mock Classes ============

class MockDirection(str, Enum):
    """模拟信号方向"""
    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class MockMarketRegime(str, Enum):
    """模拟市场状态"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class MockFusedSignal:
    """模拟融合信号"""
    direction: MockDirection = MockDirection.LONG
    strength: float = 0.7
    confidence: float = 0.8
    normalized_value: float = 0.5
    agreement_ratio: float = 0.7
    conflict_level: float = 0.2

    def to_dict(self):
        return {
            'direction': self.direction.value,
            'strength': self.strength,
            'confidence': self.confidence,
        }


@dataclass
class MockStrategySignal:
    """模拟策略信号"""
    strategy_name: str = "momentum"
    direction: MockDirection = MockDirection.LONG
    strength: float = 0.7
    confidence: float = 0.8
    reason: str = "Test reason"


@dataclass
class MockIndicatorSignal:
    """模拟指标信号"""
    indicator_name: str = "RSI"
    signal_type: str = "momentum"
    direction: MockDirection = MockDirection.LONG
    strength: float = 0.6
    confidence: float = 0.7
    reason: str = "RSI oversold"


@dataclass
class MockSentimentSignal:
    """模拟舆情信号"""
    source: str = "news"
    direction: str = "bullish"
    strength: float = 0.8

    def to_dict(self):
        return {
            'source': self.source,
            'direction': self.direction,
            'strength': self.strength,
        }


@dataclass
class MockAggregatedSignal:
    """模拟聚合信号"""
    symbol: str = "AAPL"
    fused_signal: MockFusedSignal = field(default_factory=MockFusedSignal)
    strategy_signals: List[MockStrategySignal] = field(default_factory=list)
    indicator_signals: List[MockIndicatorSignal] = field(default_factory=list)
    sentiment_signals: List[MockSentimentSignal] = field(default_factory=list)
    market_regime: MockMarketRegime = MockMarketRegime.RANGING
    sentiment_alert: Dict[str, Any] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.strategy_signals:
            self.strategy_signals = [MockStrategySignal()]
        if not self.indicator_signals:
            self.indicator_signals = [MockIndicatorSignal()]


# ============ Fixtures ============

@pytest.fixture
def bridge_config():
    """基础桥接器配置"""
    return BridgeConfig(
        min_fused_strength=0.4,
        min_fused_confidence=0.5,
        min_agreement_ratio=0.5,
        max_conflict_level=0.7,
        signal_cooldown_seconds=60,
        trade_cooldown_seconds=30,
        auto_execute=False,  # 测试时关闭自动执行
        max_signals_per_minute=10,
    )


@pytest.fixture
def bridge(bridge_config):
    """桥接器实例"""
    return SignalBridge(config=bridge_config)


@pytest.fixture
def strong_signal():
    """强信号"""
    return MockAggregatedSignal(
        symbol="AAPL",
        fused_signal=MockFusedSignal(
            direction=MockDirection.STRONG_LONG,
            strength=0.85,
            confidence=0.80,
            agreement_ratio=0.8,
            conflict_level=0.1,
        ),
    )


@pytest.fixture
def weak_signal():
    """弱信号"""
    return MockAggregatedSignal(
        symbol="WEAK",
        fused_signal=MockFusedSignal(
            direction=MockDirection.LONG,
            strength=0.3,  # 低于阈值
            confidence=0.4,  # 低于阈值
        ),
    )


@pytest.fixture
def conflicting_signal():
    """冲突信号"""
    return MockAggregatedSignal(
        symbol="CONFLICT",
        fused_signal=MockFusedSignal(
            direction=MockDirection.LONG,
            strength=0.7,
            confidence=0.7,
            agreement_ratio=0.3,  # 低一致性
            conflict_level=0.8,   # 高冲突
        ),
    )


@pytest.fixture
def mock_executor():
    """模拟执行器"""
    executor = MagicMock(spec=SignalExecutor)
    executor._execute_signal = AsyncMock(return_value=MagicMock(success=True))
    return executor


# ============ 状态管理测试 ============

class TestBridgeState:
    """桥接器状态测试"""

    def test_initial_state(self, bridge):
        """测试初始状态"""
        assert bridge.state == BridgeState.IDLE

    def test_start(self, bridge):
        """测试启动"""
        bridge.start()
        assert bridge.state == BridgeState.RUNNING

    def test_stop(self, bridge):
        """测试停止"""
        bridge.start()
        bridge.stop()
        assert bridge.state == BridgeState.IDLE

    def test_pause_resume(self, bridge):
        """测试暂停和恢复"""
        bridge.start()
        bridge.pause()
        assert bridge.state == BridgeState.PAUSED

        bridge.resume()
        assert bridge.state == BridgeState.RUNNING


# ============ 信号过滤测试 ============

class TestSignalFiltering:
    """信号过滤测试"""

    @pytest.mark.asyncio
    async def test_filter_when_not_running(self, bridge, strong_signal):
        """测试未运行时过滤"""
        result = await bridge.process_aggregated_signal(strong_signal)

        assert result.filter_result == SignalFilterResult.FILTERED_CUSTOM
        assert "not running" in result.filter_reason.lower()

    @pytest.mark.asyncio
    async def test_pass_strong_signal(self, bridge, strong_signal):
        """测试强信号通过"""
        bridge.start()
        result = await bridge.process_aggregated_signal(strong_signal)

        assert result.filter_result == SignalFilterResult.PASS

    @pytest.mark.asyncio
    async def test_filter_weak_strength(self, bridge, weak_signal):
        """测试弱强度信号被过滤"""
        bridge.start()
        result = await bridge.process_aggregated_signal(weak_signal)

        assert result.filter_result == SignalFilterResult.FILTERED_WEAK
        assert "strength" in result.filter_reason.lower()

    @pytest.mark.asyncio
    async def test_filter_low_confidence(self, bridge):
        """测试低置信度信号被过滤"""
        bridge.start()
        signal = MockAggregatedSignal(
            fused_signal=MockFusedSignal(
                strength=0.7,
                confidence=0.3,  # 低于阈值
            )
        )
        result = await bridge.process_aggregated_signal(signal)

        assert result.filter_result == SignalFilterResult.FILTERED_WEAK

    @pytest.mark.asyncio
    async def test_filter_high_conflict(self, bridge, conflicting_signal):
        """测试高冲突信号被过滤"""
        bridge.start()
        result = await bridge.process_aggregated_signal(conflicting_signal)

        assert result.filter_result == SignalFilterResult.FILTERED_CONFLICT


# ============ 冷却机制测试 ============

class TestCooldownMechanism:
    """冷却机制测试"""

    @pytest.mark.asyncio
    async def test_signal_cooldown(self, bridge_config, strong_signal):
        """测试信号冷却"""
        # 设置很短的冷却时间便于测试
        bridge_config.signal_cooldown_seconds = 0.5
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        # 第一次应该通过
        result1 = await bridge.process_aggregated_signal(strong_signal)
        assert result1.filter_result == SignalFilterResult.PASS

        # 立即发送第二个信号应该被冷却
        result2 = await bridge.process_aggregated_signal(strong_signal)
        assert result2.filter_result == SignalFilterResult.FILTERED_COOLDOWN

        # 等待冷却后应该通过
        await asyncio.sleep(0.6)
        result3 = await bridge.process_aggregated_signal(strong_signal)
        assert result3.filter_result == SignalFilterResult.PASS

    @pytest.mark.asyncio
    async def test_clear_cooldown(self, bridge, strong_signal):
        """测试清除冷却"""
        bridge.start()

        # 发送信号触发冷却
        await bridge.process_aggregated_signal(strong_signal)

        # 清除冷却
        bridge.clear_cooldowns("AAPL")

        # 应该可以再次发送
        result = await bridge.process_aggregated_signal(strong_signal)
        assert result.filter_result == SignalFilterResult.PASS

    @pytest.mark.asyncio
    async def test_clear_all_cooldowns(self, bridge):
        """测试清除所有冷却"""
        bridge.start()

        # 发送多个标的的信号
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            signal = MockAggregatedSignal(symbol=symbol)
            await bridge.process_aggregated_signal(signal)

        # 清除所有冷却
        bridge.clear_cooldowns()

        # 所有标的都应该可以再次发送
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            signal = MockAggregatedSignal(symbol=symbol)
            result = await bridge.process_aggregated_signal(signal)
            assert result.filter_result == SignalFilterResult.PASS


# ============ 速率限制测试 ============

class TestRateLimiting:
    """速率限制测试"""

    @pytest.mark.asyncio
    async def test_rate_limit(self, bridge_config):
        """测试速率限制"""
        bridge_config.max_signals_per_minute = 3
        bridge_config.signal_cooldown_seconds = 0  # 禁用冷却
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        results = []
        for i in range(5):
            signal = MockAggregatedSignal(symbol=f"SYM{i}")
            result = await bridge.process_aggregated_signal(signal)
            results.append(result)

        # 前3个应该通过，后2个被限制
        passed = [r for r in results if r.filter_result == SignalFilterResult.PASS]
        limited = [r for r in results if r.filter_result == SignalFilterResult.FILTERED_CUSTOM]

        assert len(passed) == 3
        assert len(limited) == 2


# ============ 信号转换测试 ============

class TestSignalConversion:
    """信号转换测试"""

    @pytest.mark.asyncio
    async def test_convert_to_trading_signal(self, bridge, strong_signal):
        """测试转换为TradingSignal"""
        bridge.start()
        result = await bridge.process_aggregated_signal(strong_signal)

        assert result.trading_signal is not None
        assert result.trading_signal.symbol == "AAPL"
        assert result.trading_signal.strength == 0.85
        assert result.trading_signal.confidence == 0.80

    @pytest.mark.asyncio
    async def test_direction_mapping(self, bridge):
        """测试方向映射"""
        bridge.start()

        # 测试所有方向
        directions = [
            (MockDirection.STRONG_LONG, SignalDirection.STRONG_LONG),
            (MockDirection.LONG, SignalDirection.LONG),
            (MockDirection.SHORT, SignalDirection.SHORT),
            (MockDirection.STRONG_SHORT, SignalDirection.STRONG_SHORT),
        ]

        for mock_dir, expected_dir in directions:
            signal = MockAggregatedSignal(
                symbol=f"TEST_{mock_dir.value}",
                fused_signal=MockFusedSignal(direction=mock_dir),
            )
            bridge.clear_cooldowns()
            result = await bridge.process_aggregated_signal(signal)

            if result.trading_signal:
                assert result.trading_signal.direction == expected_dir


# ============ 批量处理测试 ============

class TestBatchProcessing:
    """批量处理测试"""

    @pytest.mark.asyncio
    async def test_batch_process(self, bridge_config):
        """测试批量处理"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        signals = [
            MockAggregatedSignal(symbol=f"SYM{i}")
            for i in range(5)
        ]

        results = await bridge.process_signals_batch(signals)

        assert len(results) == 5
        passed = [r for r in results if r.filter_result == SignalFilterResult.PASS]
        assert len(passed) == 5

    @pytest.mark.asyncio
    async def test_batch_with_prices(self, bridge_config, mock_executor):
        """测试带价格的批量处理"""
        bridge_config.signal_cooldown_seconds = 0
        bridge_config.auto_execute = True
        bridge = SignalBridge(config=bridge_config, signal_executor=mock_executor)
        bridge.start()

        signals = [
            MockAggregatedSignal(symbol="AAPL"),
            MockAggregatedSignal(symbol="MSFT"),
        ]
        prices = {"AAPL": 150.0, "MSFT": 350.0}

        results = await bridge.process_signals_batch(
            signals, portfolio_id=1, prices=prices
        )

        assert len(results) == 2


# ============ 统计功能测试 ============

class TestStatistics:
    """统计功能测试"""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, bridge_config):
        """测试统计跟踪"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        # 发送一些信号
        strong = MockAggregatedSignal(symbol="STRONG")
        weak = MockAggregatedSignal(
            symbol="WEAK",
            fused_signal=MockFusedSignal(strength=0.2, confidence=0.3)
        )

        await bridge.process_aggregated_signal(strong)
        await bridge.process_aggregated_signal(weak)

        stats = bridge.get_statistics()

        assert stats['total_signals_received'] == 2
        assert stats['signals_passed'] == 1
        assert stats['signals_filtered'] == 1

    @pytest.mark.asyncio
    async def test_filter_reason_statistics(self, bridge_config):
        """测试过滤原因统计"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        # 弱强度信号
        weak_strength = MockAggregatedSignal(
            symbol="WEAK1",
            fused_signal=MockFusedSignal(strength=0.2, confidence=0.8)
        )
        # 低置信度信号
        low_conf = MockAggregatedSignal(
            symbol="WEAK2",
            fused_signal=MockFusedSignal(strength=0.8, confidence=0.3)
        )

        await bridge.process_aggregated_signal(weak_strength)
        await bridge.process_aggregated_signal(low_conf)

        stats = bridge.get_statistics()

        # 两个都应该因为强度/置信度被过滤
        assert stats['filtered_by_strength'] + stats['filtered_by_confidence'] == 2


# ============ 持仓管理测试 ============

class TestPositionManagement:
    """持仓管理测试"""

    def test_update_positions(self, bridge):
        """测试更新持仓"""
        bridge.update_positions({"AAPL", "MSFT"})
        stats = bridge.get_statistics()
        assert "AAPL" in stats['current_positions']
        assert "MSFT" in stats['current_positions']

    def test_add_remove_position(self, bridge):
        """测试添加和移除持仓"""
        bridge.add_position("AAPL")
        stats = bridge.get_statistics()
        assert "AAPL" in stats['current_positions']

        bridge.remove_position("AAPL")
        stats = bridge.get_statistics()
        assert "AAPL" not in stats['current_positions']


# ============ 历史查询测试 ============

class TestHistoryQuery:
    """历史查询测试"""

    @pytest.mark.asyncio
    async def test_signal_history(self, bridge_config):
        """测试信号历史"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        # 发送几个信号
        for i in range(5):
            signal = MockAggregatedSignal(symbol=f"SYM{i}")
            await bridge.process_aggregated_signal(signal)

        history = bridge.get_signal_history()
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_history_filter_by_symbol(self, bridge_config):
        """测试按标的过滤历史"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        await bridge.process_aggregated_signal(MockAggregatedSignal(symbol="AAPL"))
        await bridge.process_aggregated_signal(MockAggregatedSignal(symbol="MSFT"))
        await bridge.process_aggregated_signal(MockAggregatedSignal(symbol="AAPL"))

        aapl_history = bridge.get_signal_history(symbol="AAPL")
        assert len(aapl_history) == 2

    @pytest.mark.asyncio
    async def test_history_passed_only(self, bridge_config):
        """测试只获取通过的信号"""
        bridge_config.signal_cooldown_seconds = 0
        bridge = SignalBridge(config=bridge_config)
        bridge.start()

        # 一个强信号，一个弱信号
        strong = MockAggregatedSignal(symbol="STRONG")
        weak = MockAggregatedSignal(
            symbol="WEAK",
            fused_signal=MockFusedSignal(strength=0.2)
        )

        await bridge.process_aggregated_signal(strong)
        await bridge.process_aggregated_signal(weak)

        passed = bridge.get_signal_history(passed_only=True)
        assert len(passed) == 1
        assert passed[0].original_signal.symbol == "STRONG"


# ============ 工厂函数测试 ============

class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_signal_bridge(self):
        """测试工厂函数创建"""
        bridge = create_signal_bridge(
            min_strength=0.5,
            min_confidence=0.6,
            auto_execute=False,
        )

        assert bridge.config.min_fused_strength == 0.5
        assert bridge.config.min_fused_confidence == 0.6
        assert bridge.config.auto_execute is False

    def test_create_with_executor(self, mock_executor):
        """测试带执行器创建"""
        bridge = create_signal_bridge(executor=mock_executor)

        assert bridge.signal_executor == mock_executor


# ============ 回调测试 ============

class TestCallbacks:
    """回调函数测试"""

    @pytest.mark.asyncio
    async def test_on_signal_received(self, bridge_config, strong_signal):
        """测试信号接收回调"""
        received = []

        async def on_received(signal):
            received.append(signal)

        bridge = SignalBridge(
            config=bridge_config,
            on_signal_received=on_received,
        )
        bridge.start()

        await bridge.process_aggregated_signal(strong_signal)

        assert len(received) == 1
        assert received[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_on_signal_filtered(self, bridge_config, weak_signal):
        """测试信号过滤回调"""
        filtered = []

        async def on_filtered(bridged):
            filtered.append(bridged)

        bridge = SignalBridge(
            config=bridge_config,
            on_signal_filtered=on_filtered,
        )
        bridge.start()

        await bridge.process_aggregated_signal(weak_signal)

        assert len(filtered) == 1
        assert filtered[0].filter_result == SignalFilterResult.FILTERED_WEAK


# ============ 运行测试 ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
