"""
Trade Signal Aggregator 测试套件 - Comprehensive Test Suite

测试覆盖:
1. 基础功能测试 (Basic Functionality)
2. 信号管理测试 (Signal Management)
3. 权重管理测试 (Weight Management)
4. 加权平均聚合测试 (Weighted Average Aggregation)
5. 多数投票聚合测试 (Majority Voting)
6. 置信度加权聚合测试 (Confidence Weighted)
7. 贝叶斯融合测试 (Bayesian Fusion)
8. 集成方法测试 (Ensemble Method)
9. 冲突解决测试 (Conflict Resolution)
10. 信号归一化测试 (Signal Normalization)
11. 质量评估测试 (Quality Assessment)
12. 历史表现加权测试 (Historical Performance Weighting)
13. 实时流式处理测试 (Real-time Streaming)
14. 边界条件测试 (Edge Cases)
"""

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import asyncio
import statistics
import unittest
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.trade_signal_aggregator import (
    AggregatedSignal,
    AggregationMethod,
    AggregatorConfig,
    AggregatorState,
    ConflictResolutionStrategy,
    HistoricalPerformance,
    PerformanceProvider,
    SignalDirection,
    SignalQuality,
    SignalQualityMetrics,
    SignalType,
    SimulatedPerformanceProvider,
    StrategySignal,
    TradeSignalAggregator,
    create_trade_signal_aggregator,
)


# ============ Helper Functions ============

def create_test_signal(
    strategy_name: str = "TestStrategy",
    symbol: str = "BTCUSDT",
    direction: SignalDirection = SignalDirection.LONG,
    strength: float = 0.7,
    confidence: float = 0.8,
    signal_type: SignalType = SignalType.MOMENTUM,
) -> StrategySignal:
    """创建测试信号"""
    return StrategySignal(
        signal_id=f"TEST-{strategy_name}-{datetime.now().timestamp()}",
        strategy_name=strategy_name,
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        signal_type=signal_type,
        timestamp=datetime.now(),
    )


# ============ Test Classes ============

class TestBasicFunctionality(unittest.TestCase):
    """测试基础功能"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            min_signals_required=2,
            aggregation_interval=0.1,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    def test_01_aggregator_initialization(self):
        """测试聚合器初始化"""
        self.assertEqual(self.aggregator.state, AggregatorState.STOPPED)
        self.assertIsNotNone(self.aggregator.config)
        self.assertIsNotNone(self.aggregator.performance_provider)

    async def test_02_start_stop_aggregator(self):
        """测试启动和停止聚合器"""
        await self.aggregator.start()
        self.assertEqual(self.aggregator.state, AggregatorState.RUNNING)

        await self.aggregator.stop()
        self.assertEqual(self.aggregator.state, AggregatorState.STOPPED)

    async def test_03_pause_resume_aggregator(self):
        """测试暂停和恢复聚合器"""
        await self.aggregator.start()
        await self.aggregator.pause()
        self.assertEqual(self.aggregator.state, AggregatorState.PAUSED)

        await self.aggregator.resume()
        self.assertEqual(self.aggregator.state, AggregatorState.RUNNING)

        await self.aggregator.stop()

    def test_04_get_statistics(self):
        """测试获取统计信息"""
        stats = self.aggregator.get_statistics()
        self.assertIn("state", stats)
        self.assertIn("total_signals_received", stats)
        self.assertIn("total_aggregations", stats)
        self.assertIn("config", stats)

    def test_05_factory_function(self):
        """测试工厂函数"""
        aggregator = create_trade_signal_aggregator(
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            min_signals_required=3,
        )
        self.assertIsNotNone(aggregator)
        self.assertEqual(aggregator.config.aggregation_method, AggregationMethod.WEIGHTED_AVERAGE)
        self.assertEqual(aggregator.config.min_signals_required, 3)


class TestSignalManagement(unittest.TestCase):
    """测试信号管理"""

    def setUp(self):
        """测试前置"""
        self.aggregator = TradeSignalAggregator()

    async def test_06_add_signal(self):
        """测试添加信号"""
        signal = create_test_signal()
        await self.aggregator.add_signal(signal)

        stats = self.aggregator.get_statistics()
        self.assertEqual(stats["total_signals_received"], 1)

    async def test_07_add_multiple_signals(self):
        """测试添加多个信号"""
        for i in range(5):
            signal = create_test_signal(strategy_name=f"Strategy{i}")
            await self.aggregator.add_signal(signal)

        stats = self.aggregator.get_statistics()
        self.assertEqual(stats["total_signals_received"], 5)

    async def test_08_add_signals_batch(self):
        """测试批量添加信号"""
        signals = [create_test_signal(strategy_name=f"Strategy{i}") for i in range(5)]
        await self.aggregator.add_signals_batch(signals)

        stats = self.aggregator.get_statistics()
        self.assertEqual(stats["total_signals_received"], 5)

    async def test_09_validate_signal(self):
        """测试信号验证"""
        # 有效信号
        valid_signal = create_test_signal()
        self.assertTrue(self.aggregator._validate_signal(valid_signal))

        # 无效信号 (强度超出范围)
        invalid_signal = create_test_signal(strength=1.5)
        self.assertFalse(self.aggregator._validate_signal(invalid_signal))

        # 无效信号 (置信度超出范围)
        invalid_signal2 = create_test_signal(confidence=-0.1)
        self.assertFalse(self.aggregator._validate_signal(invalid_signal2))

    async def test_10_filter_signal(self):
        """测试信号过滤"""
        # 通过过滤的信号
        good_signal = create_test_signal(strength=0.7, confidence=0.8)
        self.assertTrue(self.aggregator._filter_signal(good_signal))

        # 强度过低
        weak_signal = create_test_signal(strength=0.1, confidence=0.8)
        self.assertFalse(self.aggregator._filter_signal(weak_signal))

        # 置信度过低
        low_conf_signal = create_test_signal(strength=0.7, confidence=0.2)
        self.assertFalse(self.aggregator._filter_signal(low_conf_signal))

    async def test_11_get_signal_history(self):
        """测试获取信号历史"""
        for i in range(10):
            signal = create_test_signal(strategy_name=f"Strategy{i}")
            await self.aggregator.add_signal(signal)

        history = self.aggregator.get_signal_history()
        self.assertEqual(len(history), 10)

        # 按策略过滤
        strategy_history = self.aggregator.get_signal_history(strategy_name="Strategy0")
        self.assertEqual(len(strategy_history), 1)

        # 按标的过滤
        symbol_history = self.aggregator.get_signal_history(symbol="BTCUSDT")
        self.assertEqual(len(symbol_history), 10)

    async def test_12_clear_buffer(self):
        """测试清空缓冲区"""
        signal = create_test_signal()
        await self.aggregator.add_signal(signal)

        self.aggregator.clear_buffer("BTCUSDT")

        # 验证缓冲区已清空
        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNone(result)


class TestWeightManagement(unittest.TestCase):
    """测试权重管理"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(use_historical_weighting=True)
        self.aggregator = TradeSignalAggregator(config=self.config)

    def test_13_set_strategy_weight(self):
        """测试设置策略权重"""
        self.aggregator.set_strategy_weight("Strategy1", 1.5)
        weight = self.aggregator.get_strategy_weight("Strategy1")
        self.assertEqual(weight, 1.5)

    def test_14_get_strategy_weight(self):
        """测试获取策略权重"""
        # 未设置的策略应返回默认权重
        weight = self.aggregator.get_strategy_weight("NonExistent")
        self.assertEqual(weight, 1.0)

    def test_15_get_all_strategy_weights(self):
        """测试获取所有策略权重"""
        self.aggregator.set_strategy_weight("Strategy1", 1.2)
        self.aggregator.set_strategy_weight("Strategy2", 0.8)

        weights = self.aggregator.get_strategy_weights()
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights["Strategy1"], 1.2)
        self.assertEqual(weights["Strategy2"], 0.8)

    def test_16_weight_bounds(self):
        """测试权重边界限制"""
        # 超过最大值
        self.aggregator.set_strategy_weight("Strategy1", 5.0)
        weight = self.aggregator.get_strategy_weight("Strategy1")
        self.assertLessEqual(weight, self.config.max_weight)

        # 低于最小值
        self.aggregator.set_strategy_weight("Strategy2", 0.01)
        weight = self.aggregator.get_strategy_weight("Strategy2")
        self.assertGreaterEqual(weight, self.config.min_weight)

    async def test_17_update_strategy_weight(self):
        """测试更新策略权重"""
        await self.aggregator._update_strategy_weight("Strategy1")
        weight = self.aggregator.get_strategy_weight("Strategy1")
        self.assertGreater(weight, 0.0)


class TestWeightedAverageAggregation(unittest.TestCase):
    """测试加权平均聚合"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            min_signals_required=2,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_18_simple_weighted_average(self):
        """测试简单加权平均"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.7, confidence=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertGreater(result.strength, 0.0)
        self.assertGreater(result.confidence, 0.0)

    async def test_19_weighted_average_with_different_weights(self):
        """测试不同权重的加权平均"""
        self.aggregator.set_strategy_weight("Strategy1", 2.0)
        self.aggregator.set_strategy_weight("Strategy2", 1.0)

        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.6),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 权重更高的策略应该有更大影响
        self.assertGreater(result.strength, 0.5)

    async def test_20_weighted_average_all_directions(self):
        """测试所有方向的加权平均"""
        directions = [
            SignalDirection.STRONG_LONG,
            SignalDirection.LONG,
            SignalDirection.NEUTRAL,
            SignalDirection.SHORT,
            SignalDirection.STRONG_SHORT,
        ]

        for i, direction in enumerate(directions):
            # Clear aggregator for each test
            aggregator = TradeSignalAggregator(config=self.config)

            signals = [
                create_test_signal(f"Strategy{i}", direction=direction, strength=0.8, symbol=f"TEST{i}"),
                create_test_signal(f"Strategy{i+5}", direction=direction, strength=0.7, symbol=f"TEST{i}"),
            ]

            for signal in signals:
                await aggregator.add_signal(signal)

            result = await aggregator.aggregate_signals(f"TEST{i}")
            self.assertIsNotNone(result)


class TestMajorityVoting(unittest.TestCase):
    """测试多数投票"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_method=AggregationMethod.MAJORITY_VOTING,
            min_signals_required=3,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_21_clear_majority(self):
        """测试明确多数"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG),
            create_test_signal("Strategy2", direction=SignalDirection.LONG),
            create_test_signal("Strategy3", direction=SignalDirection.SHORT),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertAlmostEqual(result.agreement_ratio, 2.0 / 3.0, places=2)

    async def test_22_unanimous_vote(self):
        """测试一致投票"""
        signals = [
            create_test_signal(f"Strategy{i}", direction=SignalDirection.LONG)
            for i in range(5)
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertEqual(result.agreement_ratio, 1.0)

    async def test_23_split_vote(self):
        """测试分裂投票"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG),
            create_test_signal("Strategy2", direction=SignalDirection.LONG),
            create_test_signal("Strategy3", direction=SignalDirection.SHORT),
            create_test_signal("Strategy4", direction=SignalDirection.SHORT),
            create_test_signal("Strategy5", direction=SignalDirection.NEUTRAL),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 应该选择票数最多的方向
        self.assertIn(result.direction, [SignalDirection.LONG, SignalDirection.SHORT])


class TestConfidenceWeighted(unittest.TestCase):
    """测试置信度加权"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_method=AggregationMethod.CONFIDENCE_WEIGHTED,
            min_signals_required=2,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_24_high_confidence_dominates(self):
        """测试高置信度主导"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.9, confidence=0.95),
            create_test_signal("Strategy2", direction=SignalDirection.SHORT, strength=0.7, confidence=0.5),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 高置信度的LONG信号应该占主导
        self.assertEqual(result.direction, SignalDirection.LONG)

    async def test_25_equal_confidence(self):
        """测试相等置信度"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, confidence=0.8),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, confidence=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)

    async def test_26_confidence_weighted_calculation(self):
        """测试置信度加权计算"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.6, confidence=0.7),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 验证加权后的置信度
        self.assertGreater(result.confidence, 0.4)


class TestBayesianFusion(unittest.TestCase):
    """测试贝叶斯融合"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_method=AggregationMethod.BAYESIAN_FUSION,
            min_signals_required=2,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_27_bayesian_fusion_basic(self):
        """测试基本贝叶斯融合"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.7, confidence=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)

    async def test_28_bayesian_fusion_conflicting(self):
        """测试冲突信号的贝叶斯融合"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.SHORT, strength=0.7, confidence=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 应该选择概率更高的方向

    async def test_29_bayesian_multiple_signals(self):
        """测试多信号贝叶斯融合"""
        signals = [
            create_test_signal(f"Strategy{i}", direction=SignalDirection.LONG, strength=0.7, confidence=0.8)
            for i in range(5)
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.LONG)


class TestEnsembleMethod(unittest.TestCase):
    """测试集成方法"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_method=AggregationMethod.ENSEMBLE,
            min_signals_required=2,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_30_ensemble_aggregation(self):
        """测试集成聚合"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.7, confidence=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.aggregation_method, AggregationMethod.ENSEMBLE)

    async def test_31_ensemble_best_method(self):
        """测试集成方法选择最佳结果"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.9, confidence=0.95),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy3", direction=SignalDirection.LONG, strength=0.7, confidence=0.85),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 集成方法应该产生较高置信度结果
        self.assertGreater(result.confidence, 0.4)


class TestConflictResolution(unittest.TestCase):
    """测试冲突解决"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            min_signals_required=2,
            conflict_threshold=0.4,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_32_detect_conflict(self):
        """测试检测冲突"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8),
            create_test_signal("Strategy2", direction=SignalDirection.SHORT, strength=0.7),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 应该检测到冲突
        self.assertGreater(result.conflict_level, 0.0)

    async def test_33_no_conflict(self):
        """测试无冲突情况"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.7),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 不应该有冲突
        self.assertEqual(result.conflict_level, 0.0)

    async def test_34_high_conflict(self):
        """测试高冲突情况"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.8),
            create_test_signal("Strategy3", direction=SignalDirection.SHORT, strength=0.9),
            create_test_signal("Strategy4", direction=SignalDirection.SHORT, strength=0.8),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        # 应该有明显的冲突
        self.assertGreater(result.conflict_level, 0.3)


class TestSignalNormalization(unittest.TestCase):
    """测试信号归一化"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            normalize_strength=True,
            normalize_confidence=True,
            min_signals_required=2,
        )
        self.aggregator = TradeSignalAggregator(config=self.config)

    async def test_35_normalize_strength(self):
        """测试强度归一化"""
        signals = [
            create_test_signal("Strategy1", strength=0.9),
            create_test_signal("Strategy2", strength=0.3),
            create_test_signal("Strategy3", strength=0.6),
        ]

        self.aggregator._normalize_strength(signals)

        # 归一化后应该在 [0, 1] 范围内
        for signal in signals:
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)

    async def test_36_normalize_confidence(self):
        """测试置信度归一化"""
        signals = [
            create_test_signal("Strategy1", confidence=0.9),
            create_test_signal("Strategy2", confidence=0.4),
            create_test_signal("Strategy3", confidence=0.7),
        ]

        self.aggregator._normalize_confidence(signals)

        # 归一化后应该在 [0, 1] 范围内
        for signal in signals:
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)

    async def test_37_normalization_with_aggregation(self):
        """测试归一化与聚合结合"""
        signals = [
            create_test_signal("Strategy1", strength=0.9, confidence=0.95),
            create_test_signal("Strategy2", strength=0.3, confidence=0.5),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)


class TestQualityAssessment(unittest.TestCase):
    """测试质量评估"""

    def setUp(self):
        """测试前置"""
        self.aggregator = TradeSignalAggregator()

    def test_38_assess_excellent_quality(self):
        """测试优秀质量评估"""
        quality = self.aggregator._assess_signal_quality(
            strength=0.9,
            confidence=0.95,
            agreement_ratio=0.9
        )
        self.assertEqual(quality, SignalQuality.EXCELLENT)

    def test_39_assess_good_quality(self):
        """测试良好质量评估"""
        quality = self.aggregator._assess_signal_quality(
            strength=0.7,
            confidence=0.75,
            agreement_ratio=0.7
        )
        self.assertEqual(quality, SignalQuality.GOOD)

    def test_40_assess_poor_quality(self):
        """测试较差质量评估"""
        quality = self.aggregator._assess_signal_quality(
            strength=0.3,
            confidence=0.3,
            agreement_ratio=0.3
        )
        self.assertEqual(quality, SignalQuality.POOR)

    async def test_41_quality_metrics_calculation(self):
        """测试质量指标计算"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG, strength=0.8, confidence=0.9),
            create_test_signal("Strategy2", direction=SignalDirection.LONG, strength=0.7, confidence=0.8),
            create_test_signal("Strategy3", direction=SignalDirection.SHORT, strength=0.6, confidence=0.7),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)

        metrics = self.aggregator.get_quality_metrics("BTCUSDT")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_signals, 3)
        self.assertGreater(metrics.overall_quality, 0.0)

    async def test_42_get_all_quality_metrics(self):
        """测试获取所有质量指标"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            signals = [
                create_test_signal("Strategy1", symbol=symbol),
                create_test_signal("Strategy2", symbol=symbol),
            ]
            for signal in signals:
                await self.aggregator.add_signal(signal)
            await self.aggregator.aggregate_signals(symbol)

        all_metrics = self.aggregator.get_all_quality_metrics()
        self.assertEqual(len(all_metrics), 3)


class TestHistoricalPerformanceWeighting(unittest.TestCase):
    """测试历史表现加权"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(use_historical_weighting=True)
        self.provider = SimulatedPerformanceProvider()
        self.aggregator = TradeSignalAggregator(
            config=self.config,
            performance_provider=self.provider
        )

    async def test_43_performance_weight_calculation(self):
        """测试表现权重计算"""
        # 设置历史表现
        perf = HistoricalPerformance(
            strategy_name="Strategy1",
            total_signals=100,
            successful_signals=80,
            win_rate=0.8,
            average_return=0.05,
            sharpe_ratio=1.5,
        )
        self.provider.set_performance("Strategy1", perf)

        await self.aggregator._update_strategy_weight("Strategy1")
        weight = self.aggregator.get_strategy_weight("Strategy1")

        # 好的表现应该获得正常权重
        self.assertGreater(weight, 0.5)

    async def test_44_poor_performance_weight(self):
        """测试差表现的权重"""
        perf = HistoricalPerformance(
            strategy_name="Strategy2",
            total_signals=100,
            successful_signals=30,
            win_rate=0.3,
            average_return=-0.02,
            sharpe_ratio=-0.5,
        )
        self.provider.set_performance("Strategy2", perf)

        await self.aggregator._update_strategy_weight("Strategy2")
        weight = self.aggregator.get_strategy_weight("Strategy2")

        # 差的表现应该获得较低权重
        self.assertLess(weight, 1.0)

    async def test_45_update_performance(self):
        """测试更新表现"""
        await self.provider.update_performance("Strategy1", success=True, return_value=0.05)
        perf = await self.provider.get_performance("Strategy1")

        self.assertIsNotNone(perf)
        self.assertEqual(perf.total_signals, 1)
        self.assertEqual(perf.successful_signals, 1)

    def test_46_performance_to_dict(self):
        """测试表现序列化"""
        perf = HistoricalPerformance(
            strategy_name="Strategy1",
            total_signals=100,
            win_rate=0.75,
        )

        data = perf.to_dict()
        self.assertIn("strategy_name", data)
        self.assertIn("total_signals", data)
        self.assertIn("win_rate", data)


class TestRealtimeStreaming(unittest.TestCase):
    """测试实时流式处理"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig(
            aggregation_interval=0.1,
            min_signals_required=2,
        )
        self.aggregations = []

        def aggregation_callback(agg: AggregatedSignal):
            self.aggregations.append(agg)

        self.aggregator = TradeSignalAggregator(
            config=self.config,
            aggregation_callback=aggregation_callback,
        )

    async def test_47_streaming_aggregation(self):
        """测试流式聚合"""
        await self.aggregator.start()

        # 添加信号
        for i in range(4):
            signal = create_test_signal(f"Strategy{i}")
            await self.aggregator.add_signal(signal)

        # 等待处理
        await asyncio.sleep(0.3)

        await self.aggregator.stop()

        # 验证回调被触发
        self.assertGreater(len(self.aggregations), 0)

    async def test_48_continuous_streaming(self):
        """测试连续流式处理"""
        await self.aggregator.start()

        # 持续添加信号
        for batch in range(3):
            for i in range(2):
                signal = create_test_signal(f"Strategy{i}", symbol=f"TEST{batch}")
                await self.aggregator.add_signal(signal)
            await asyncio.sleep(0.15)

        await self.aggregator.stop()

        # 应该有多次聚合
        self.assertGreaterEqual(len(self.aggregations), 2)

    async def test_49_get_aggregation_history(self):
        """测试获取聚合历史"""
        signals = [
            create_test_signal("Strategy1"),
            create_test_signal("Strategy2"),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        await self.aggregator.aggregate_signals("BTCUSDT")

        history = self.aggregator.get_aggregation_history()
        self.assertEqual(len(history), 1)

        # 按标的过滤
        symbol_history = self.aggregator.get_aggregation_history(symbol="BTCUSDT")
        self.assertEqual(len(symbol_history), 1)


class TestEdgeCases(unittest.TestCase):
    """测试边界条件"""

    def setUp(self):
        """测试前置"""
        self.aggregator = TradeSignalAggregator()

    async def test_50_insufficient_signals(self):
        """测试信号不足"""
        signal = create_test_signal()
        await self.aggregator.add_signal(signal)

        # 只有一个信号，不满足最小要求
        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNone(result)

    async def test_51_empty_buffer(self):
        """测试空缓冲区"""
        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNone(result)

    async def test_52_zero_strength_signals(self):
        """测试零强度信号"""
        signals = [
            create_test_signal("Strategy1", strength=0.0),
            create_test_signal("Strategy2", strength=0.0),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        # 零强度信号应该被过滤
        self.assertIsNone(result)

    async def test_53_all_neutral_signals(self):
        """测试全中性信号"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.NEUTRAL),
            create_test_signal("Strategy2", direction=SignalDirection.NEUTRAL),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.direction, SignalDirection.NEUTRAL)

    async def test_54_single_signal_type(self):
        """测试单一信号类型"""
        signals = [
            create_test_signal("Strategy1", signal_type=SignalType.MOMENTUM),
            create_test_signal("Strategy2", signal_type=SignalType.MOMENTUM),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)

    async def test_55_diverse_signal_types(self):
        """测试多样化信号类型"""
        signals = [
            create_test_signal("Strategy1", signal_type=SignalType.MOMENTUM),
            create_test_signal("Strategy2", signal_type=SignalType.TREND_FOLLOWING),
            create_test_signal("Strategy3", signal_type=SignalType.MEAN_REVERSION),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)

        metrics = self.aggregator.get_quality_metrics("BTCUSDT")
        if metrics:
            # 多样性分数应该较高
            self.assertGreater(metrics.diversity_score, 0.0)

    async def test_56_very_high_confidence(self):
        """测试极高置信度"""
        signals = [
            create_test_signal("Strategy1", confidence=0.99, strength=0.9),
            create_test_signal("Strategy2", confidence=0.98, strength=0.9),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.4)
        # High confidence and high strength should produce good or excellent quality
        self.assertIn(result.quality, [SignalQuality.GOOD, SignalQuality.EXCELLENT])

    async def test_57_very_low_confidence(self):
        """测试极低置信度"""
        config = AggregatorConfig(
            min_signal_confidence=0.0,  # 允许低置信度信号
            require_minimum_quality=False,
        )
        aggregator = TradeSignalAggregator(config=config)

        signals = [
            create_test_signal("Strategy1", confidence=0.1),
            create_test_signal("Strategy2", confidence=0.1),
        ]

        for signal in signals:
            await aggregator.add_signal(signal)

        result = await aggregator.aggregate_signals("BTCUSDT")
        if result:
            self.assertLess(result.confidence, 0.3)

    async def test_58_mixed_strength_signals(self):
        """测试混合强度信号"""
        signals = [
            create_test_signal("Strategy1", strength=0.9),
            create_test_signal("Strategy2", strength=0.5),
            create_test_signal("Strategy3", strength=0.3),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)

    async def test_59_many_signals(self):
        """测试大量信号"""
        signals = [
            create_test_signal(f"Strategy{i}", direction=SignalDirection.LONG)
            for i in range(20)
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.signal_count, 20)

    async def test_60_signal_to_dict(self):
        """测试信号序列化"""
        signal = create_test_signal()
        data = signal.to_dict()

        self.assertIn("signal_id", data)
        self.assertIn("strategy_name", data)
        self.assertIn("symbol", data)
        self.assertIn("direction", data)

    async def test_61_aggregated_signal_to_dict(self):
        """测试聚合信号序列化"""
        signals = [
            create_test_signal("Strategy1"),
            create_test_signal("Strategy2"),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        result = await self.aggregator.aggregate_signals("BTCUSDT")
        self.assertIsNotNone(result)

        data = result.to_dict()
        self.assertIn("aggregation_id", data)
        self.assertIn("symbol", data)
        self.assertIn("direction", data)
        self.assertIn("quality", data)
        self.assertIn("input_signals", data)

    async def test_62_quality_metrics_to_dict(self):
        """测试质量指标序列化"""
        signals = [
            create_test_signal("Strategy1"),
            create_test_signal("Strategy2"),
        ]

        for signal in signals:
            await self.aggregator.add_signal(signal)

        await self.aggregator.aggregate_signals("BTCUSDT")
        metrics = self.aggregator.get_quality_metrics("BTCUSDT")

        if metrics:
            data = metrics.to_dict()
            self.assertIn("symbol", data)
            self.assertIn("overall_quality", data)
            self.assertIn("consistency_score", data)

    def test_63_config_to_dict(self):
        """测试配置序列化"""
        config = AggregatorConfig()
        data = config.to_dict()

        self.assertIn("aggregation_method", data)
        self.assertIn("min_signals_required", data)
        self.assertIn("use_historical_weighting", data)

    async def test_64_multiple_symbols(self):
        """测试多标的"""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            signals = [
                create_test_signal("Strategy1", symbol=symbol),
                create_test_signal("Strategy2", symbol=symbol),
            ]
            for signal in signals:
                await self.aggregator.add_signal(signal)

        # 分别聚合每个标的
        results = []
        for symbol in symbols:
            result = await self.aggregator.aggregate_signals(symbol)
            if result:
                results.append(result)

        self.assertEqual(len(results), 3)

    async def test_65_agreement_ratio_calculation(self):
        """测试一致性比率计算"""
        signals = [
            create_test_signal("Strategy1", direction=SignalDirection.LONG),
            create_test_signal("Strategy2", direction=SignalDirection.LONG),
            create_test_signal("Strategy3", direction=SignalDirection.LONG),
            create_test_signal("Strategy4", direction=SignalDirection.SHORT),
        ]

        agreement_ratio, conflict_level = self.aggregator._calculate_agreement_metrics(signals)

        # 3/4 = 0.75
        self.assertAlmostEqual(agreement_ratio, 0.75, places=2)
        # 应该有冲突
        self.assertGreater(conflict_level, 0.0)


# ============ Test Runner ============

def run_async_test(coro):
    """运行异步测试"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# 将异步测试包装为同步测试
for test_class in [
    TestBasicFunctionality,
    TestSignalManagement,
    TestWeightManagement,
    TestWeightedAverageAggregation,
    TestMajorityVoting,
    TestConfidenceWeighted,
    TestBayesianFusion,
    TestEnsembleMethod,
    TestConflictResolution,
    TestSignalNormalization,
    TestQualityAssessment,
    TestHistoricalPerformanceWeighting,
    TestRealtimeStreaming,
    TestEdgeCases,
]:
    for attr_name in dir(test_class):
        attr = getattr(test_class, attr_name)
        if callable(attr) and attr_name.startswith('test_') and asyncio.iscoroutinefunction(attr):
            # 包装异步测试
            def make_sync_test(async_test):
                def sync_test(self):
                    return run_async_test(async_test(self))
                return sync_test

            setattr(test_class, attr_name, make_sync_test(attr))


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
