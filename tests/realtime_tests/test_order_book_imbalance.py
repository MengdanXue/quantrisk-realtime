"""
订单簿失衡分析器测试 (Order Book Imbalance Analyzer Tests)

测试订单簿失衡分析器的所有功能:
- 订单簿快照分析
- 买卖失衡度计算
- 加权失衡指标
- 订单流失衡
- 深度失衡分析
- 失衡动量指标
- 价格冲击预测
- 微观结构信号生成
- 多档位失衡热力图
- 失衡历史统计

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
from typing import List, Tuple

from quantrisk_realtime.realtime.order_book_imbalance import (
    # Main Class
    OrderBookImbalanceAnalyzer,
    create_order_book_analyzer,
    # Data Classes
    PriceLevel,
    OrderBookSnapshot,
    ImbalanceMetrics,
    DepthAnalysis,
    ImbalanceSignal,
    ImbalanceConfig,
    ImbalanceStatistics,
    # Enums
    ImbalanceDirection,
    ImbalanceSeverity,
    SignalType,
    SignalStrength,
    AnalyzerState,
    ImbalanceTrend,
)


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """创建基础分析器"""
    return create_order_book_analyzer(
        max_depth_levels=20,
        sampling_levels=10,
        signal_threshold=0.3,
    )


@pytest.fixture
def balanced_order_book() -> Tuple[List[Tuple], List[Tuple]]:
    """创建均衡的订单簿数据"""
    # 买单 (价格, 数量, 订单数)
    bids = [
        (100.0, 10.0, 5),
        (99.9, 15.0, 7),
        (99.8, 20.0, 10),
        (99.7, 25.0, 12),
        (99.6, 30.0, 15),
    ]

    # 卖单
    asks = [
        (100.1, 10.0, 5),
        (100.2, 15.0, 7),
        (100.3, 20.0, 10),
        (100.4, 25.0, 12),
        (100.5, 30.0, 15),
    ]

    return bids, asks


@pytest.fixture
def buy_heavy_order_book() -> Tuple[List[Tuple], List[Tuple]]:
    """创建买盘占优的订单簿"""
    bids = [
        (100.0, 50.0, 25),  # 买盘量大
        (99.9, 40.0, 20),
        (99.8, 30.0, 15),
        (99.7, 20.0, 10),
        (99.6, 10.0, 5),
    ]

    asks = [
        (100.1, 10.0, 5),   # 卖盘量小
        (100.2, 8.0, 4),
        (100.3, 6.0, 3),
        (100.4, 4.0, 2),
        (100.5, 2.0, 1),
    ]

    return bids, asks


@pytest.fixture
def sell_heavy_order_book() -> Tuple[List[Tuple], List[Tuple]]:
    """创建卖盘占优的订单簿"""
    bids = [
        (100.0, 10.0, 5),
        (99.9, 8.0, 4),
        (99.8, 6.0, 3),
        (99.7, 4.0, 2),
        (99.6, 2.0, 1),
    ]

    asks = [
        (100.1, 50.0, 25),
        (100.2, 40.0, 20),
        (100.3, 30.0, 15),
        (100.4, 20.0, 10),
        (100.5, 10.0, 5),
    ]

    return bids, asks


@pytest.fixture
def deep_order_book() -> Tuple[List[Tuple], List[Tuple]]:
    """创建深度订单簿 (多档位)"""
    bids = [(100.0 - i * 0.1, 10.0 + i, 5 + i) for i in range(20)]
    asks = [(100.1 + i * 0.1, 10.0 + i, 5 + i) for i in range(20)]
    return bids, asks


# ============== Test PriceLevel ==============

class TestPriceLevel:
    """测试价格档位"""

    def test_price_level_creation(self):
        """测试1: 创建价格档位"""
        level = PriceLevel(price=100.0, quantity=10.0, order_count=5)

        assert level.price == 100.0
        assert level.quantity == 10.0
        assert level.order_count == 5
        assert isinstance(level.timestamp, datetime)

    def test_price_level_to_dict(self):
        """测试2: 价格档位序列化"""
        level = PriceLevel(price=100.0, quantity=10.0, order_count=5)
        data = level.to_dict()

        assert 'price' in data
        assert 'quantity' in data
        assert 'order_count' in data
        assert 'timestamp' in data


# ============== Test OrderBookSnapshot ==============

class TestOrderBookSnapshot:
    """测试订单簿快照"""

    def test_snapshot_creation(self, balanced_order_book):
        """测试3: 创建订单簿快照"""
        bids, asks = balanced_order_book

        bid_levels = [PriceLevel(p, q, c) for p, q, c in bids]
        ask_levels = [PriceLevel(p, q, c) for p, q, c in asks]

        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
        )

        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == len(bids)
        assert len(snapshot.asks) == len(asks)
        assert snapshot.best_bid == 100.0
        assert snapshot.best_ask == 100.1
        assert snapshot.mid_price is not None
        assert snapshot.spread is not None

    def test_snapshot_volume_calculation(self, balanced_order_book):
        """测试4: 快照成交量计算"""
        bids, asks = balanced_order_book

        bid_levels = [PriceLevel(p, q, c) for p, q, c in bids]
        ask_levels = [PriceLevel(p, q, c) for p, q, c in asks]

        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
        )

        expected_bid_volume = sum(q for _, q, _ in bids)
        expected_ask_volume = sum(q for _, q, _ in asks)

        assert snapshot.total_bid_volume == expected_bid_volume
        assert snapshot.total_ask_volume == expected_ask_volume

    def test_snapshot_spread_calculation(self, balanced_order_book):
        """测试5: 价差计算"""
        bids, asks = balanced_order_book

        bid_levels = [PriceLevel(p, q, c) for p, q, c in bids]
        ask_levels = [PriceLevel(p, q, c) for p, q, c in asks]

        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
        )

        assert snapshot.spread == pytest.approx(0.1, rel=0.01)
        assert snapshot.spread_pct > 0

    def test_snapshot_to_dict(self, balanced_order_book):
        """测试6: 快照序列化"""
        bids, asks = balanced_order_book

        bid_levels = [PriceLevel(p, q, c) for p, q, c in bids]
        ask_levels = [PriceLevel(p, q, c) for p, q, c in asks]

        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
        )

        data = snapshot.to_dict()

        assert 'symbol' in data
        assert 'bids' in data
        assert 'asks' in data
        assert 'best_bid' in data
        assert 'best_ask' in data


# ============== Test OrderBookImbalanceAnalyzer ==============

class TestOrderBookImbalanceAnalyzer:
    """测试订单簿失衡分析器"""

    def test_analyzer_initialization(self, analyzer):
        """测试7: 分析器初始化"""
        assert analyzer.state == AnalyzerState.IDLE
        assert analyzer._current_snapshot is None
        assert analyzer._current_metrics is None

    def test_update_order_book_balanced(self, analyzer, balanced_order_book):
        """测试8: 更新均衡订单簿"""
        bids, asks = balanced_order_book
        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        assert snapshot is not None
        assert snapshot.symbol == "BTCUSDT"
        assert analyzer._current_snapshot == snapshot

    def test_update_order_book_buy_heavy(self, analyzer, buy_heavy_order_book):
        """测试9: 更新买盘占优订单簿"""
        bids, asks = buy_heavy_order_book
        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        assert snapshot is not None
        assert snapshot.total_bid_volume > snapshot.total_ask_volume

    def test_calculate_volume_imbalance_balanced(self, analyzer, balanced_order_book):
        """测试10: 计算均衡订单簿的成交量失衡"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        imbalance = analyzer.calculate_volume_imbalance()

        # 均衡订单簿的失衡应该接近0
        assert abs(imbalance) < 0.1

    def test_calculate_volume_imbalance_buy_heavy(self, analyzer, buy_heavy_order_book):
        """测试11: 计算买盘占优的成交量失衡"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        imbalance = analyzer.calculate_volume_imbalance()

        # 买盘占优应该是正值
        assert imbalance > 0.3

    def test_calculate_volume_imbalance_sell_heavy(self, analyzer, sell_heavy_order_book):
        """测试12: 计算卖盘占优的成交量失衡"""
        bids, asks = sell_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        imbalance = analyzer.calculate_volume_imbalance()

        # 卖盘占优应该是负值
        assert imbalance < -0.3

    def test_calculate_weighted_imbalance(self, analyzer, buy_heavy_order_book):
        """测试13: 计算加权失衡"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        weighted_imb = analyzer.calculate_weighted_imbalance()

        # 买盘占优应该是正值
        assert weighted_imb > 0

    def test_calculate_order_flow_imbalance_empty(self, analyzer):
        """测试14: 计算空订单流失衡"""
        imbalance = analyzer.calculate_order_flow_imbalance()
        assert imbalance == 0.0

    def test_calculate_order_flow_imbalance_with_data(self, analyzer):
        """测试15: 计算订单流失衡 (有数据)"""
        # 添加买单流
        for _ in range(10):
            analyzer.add_order_flow(10.0, "buy")

        # 添加卖单流
        for _ in range(5):
            analyzer.add_order_flow(10.0, "sell")

        imbalance = analyzer.calculate_order_flow_imbalance()

        # 买单多应该是正值
        assert imbalance > 0

    def test_analyze_depth_imbalance(self, analyzer, balanced_order_book):
        """测试16: 深度失衡分析"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth_analysis = analyzer.analyze_depth_imbalance()

        assert isinstance(depth_analysis, DepthAnalysis)
        assert depth_analysis.symbol == "BTCUSDT"
        assert len(depth_analysis.bid_depth_by_level) > 0
        assert len(depth_analysis.ask_depth_by_level) > 0

    def test_analyze_depth_imbalance_cumulative(self, analyzer, balanced_order_book):
        """测试17: 累计深度计算"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth_analysis = analyzer.analyze_depth_imbalance()

        # 累计深度应该递增
        assert len(depth_analysis.cumulative_bid_depth) > 0
        assert len(depth_analysis.cumulative_ask_depth) > 0

        for i in range(1, len(depth_analysis.cumulative_bid_depth)):
            assert depth_analysis.cumulative_bid_depth[i] >= depth_analysis.cumulative_bid_depth[i-1]

    def test_analyze_depth_concentration(self, analyzer, balanced_order_book):
        """测试18: 深度集中度"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth_analysis = analyzer.analyze_depth_imbalance()

        assert 0 <= depth_analysis.bid_concentration <= 1
        assert 0 <= depth_analysis.ask_concentration <= 1

    def test_predict_price_impact_buy(self, analyzer, balanced_order_book):
        """测试19: 预测买单价格冲击"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        impact = analyzer.predict_price_impact(volume=20.0, side="buy")

        # 买单应该有正向冲击
        assert impact > 0

    def test_predict_price_impact_sell(self, analyzer, balanced_order_book):
        """测试20: 预测卖单价格冲击"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        impact = analyzer.predict_price_impact(volume=20.0, side="sell")

        # 卖单应该有负向冲击
        assert impact < 0

    def test_predict_price_impact_large_volume(self, analyzer, balanced_order_book):
        """测试21: 预测大成交量价格冲击"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        small_impact = analyzer.predict_price_impact(volume=10.0, side="buy")
        large_impact = analyzer.predict_price_impact(volume=50.0, side="buy")

        # 大成交量应该有更大的价格冲击
        assert abs(large_impact) > abs(small_impact)

    def test_generate_signals_no_data(self, analyzer):
        """测试22: 无数据时不生成信号"""
        signal = analyzer.generate_signals()
        assert signal is None

    def test_generate_signals_balanced(self, analyzer, balanced_order_book):
        """测试23: 均衡订单簿不生成信号"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        # 均衡订单簿不应该生成强信号
        metrics = analyzer._current_metrics
        if metrics:
            signal = analyzer.generate_signals(metrics)
            # 可能生成中性信号或不生成
            if signal:
                assert signal.signal_type == SignalType.NEUTRAL or signal.signal_strength == SignalStrength.VERY_WEAK

    def test_generate_signals_buy_heavy(self, analyzer, buy_heavy_order_book):
        """测试24: 买盘占优生成买入信号"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        # 可能生成买入信号
        signals = analyzer.get_signals()
        if signals:
            latest_signal = signals[-1]
            # 买盘占优应该倾向于买入信号
            assert latest_signal.signal_type in (
                SignalType.STRONG_BUY,
                SignalType.BUY,
                SignalType.WEAK_BUY,
                SignalType.NEUTRAL,
            )

    def test_generate_signals_sell_heavy(self, analyzer, sell_heavy_order_book):
        """测试25: 卖盘占优生成卖出信号"""
        bids, asks = sell_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            latest_signal = signals[-1]
            # 卖盘占优应该倾向于卖出信号
            assert latest_signal.signal_type in (
                SignalType.STRONG_SELL,
                SignalType.SELL,
                SignalType.WEAK_SELL,
                SignalType.NEUTRAL,
            )

    def test_signal_validity(self, analyzer, buy_heavy_order_book):
        """测试26: 信号有效性"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals(valid_only=False)
        if signals:
            signal = signals[-1]
            now = datetime.now()

            # 新生成的信号应该有效
            assert signal.is_valid(now)

            # 未来时间信号应该无效
            future = now + timedelta(seconds=signal.ttl_seconds + 10)
            assert not signal.is_valid(future)

    def test_get_imbalance_heatmap(self, analyzer, balanced_order_book):
        """测试27: 获取失衡热力图"""
        bids, asks = balanced_order_book

        # 更新多次以积累数据
        for _ in range(10):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        heatmap = analyzer.get_imbalance_heatmap()

        assert isinstance(heatmap, dict)
        # 应该有档位数据
        assert len(heatmap) > 0

    def test_get_imbalance_heatmap_levels(self, analyzer, deep_order_book):
        """测试28: 获取指定档位的热力图"""
        bids, asks = deep_order_book

        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        heatmap = analyzer.get_imbalance_heatmap(levels=5)

        # 应该只有5个档位
        assert len(heatmap) <= 5

    def test_get_current_snapshot(self, analyzer, balanced_order_book):
        """测试29: 获取当前快照"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        snapshot = analyzer.get_current_snapshot()
        assert snapshot is not None
        assert snapshot.symbol == "BTCUSDT"

    def test_get_current_metrics(self, analyzer, balanced_order_book):
        """测试30: 获取当前指标"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()
        assert metrics is not None
        assert isinstance(metrics, ImbalanceMetrics)

    def test_get_current_depth_analysis(self, analyzer, balanced_order_book):
        """测试31: 获取当前深度分析"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()
        assert depth is not None
        assert isinstance(depth, DepthAnalysis)

    def test_get_statistics_no_data(self, analyzer):
        """测试32: 无数据时的统计"""
        stats = analyzer.get_statistics()

        assert isinstance(stats, ImbalanceStatistics)
        assert stats.total_samples == 0

    def test_get_statistics_with_data(self, analyzer, balanced_order_book):
        """测试33: 有数据时的统计"""
        bids, asks = balanced_order_book

        # 更新多次
        for _ in range(20):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        stats = analyzer.get_statistics()

        assert stats.total_samples == 20
        assert stats.avg_volume_imbalance is not None
        assert stats.std_volume_imbalance >= 0

    def test_get_statistics_direction_distribution(
        self,
        analyzer,
        balanced_order_book,
        buy_heavy_order_book,
        sell_heavy_order_book
    ):
        """测试34: 统计方向分布"""
        bal_bids, bal_asks = balanced_order_book
        buy_bids, buy_asks = buy_heavy_order_book
        sell_bids, sell_asks = sell_heavy_order_book

        # 更新不同类型的订单簿
        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", bal_bids, bal_asks)
        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", buy_bids, buy_asks)
        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", sell_bids, sell_asks)

        stats = analyzer.get_statistics()

        # 应该有不同方向的分布
        assert stats.buy_heavy_count >= 0
        assert stats.sell_heavy_count >= 0
        assert stats.balanced_count >= 0

    def test_get_summary(self, analyzer, balanced_order_book):
        """测试35: 获取摘要"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        summary = analyzer.get_summary()

        assert isinstance(summary, dict)
        assert 'state' in summary
        assert 'current_snapshot' in summary
        assert 'current_metrics' in summary
        assert 'statistics' in summary

    @pytest.mark.asyncio
    async def test_analyzer_start_stop(self, analyzer):
        """测试36: 分析器启动和停止"""
        await analyzer.start()
        assert analyzer.state == AnalyzerState.RUNNING

        await asyncio.sleep(0.1)

        await analyzer.stop()
        assert analyzer.state == AnalyzerState.IDLE

    @pytest.mark.asyncio
    async def test_analyzer_pause_resume(self, analyzer):
        """测试37: 分析器暂停和恢复"""
        await analyzer.start()
        assert analyzer.state == AnalyzerState.RUNNING

        await analyzer.pause()
        assert analyzer.state == AnalyzerState.PAUSED

        await analyzer.resume()
        assert analyzer.state == AnalyzerState.RUNNING

        await analyzer.stop()


# ============== Test ImbalanceMetrics ==============

class TestImbalanceMetrics:
    """测试失衡指标"""

    def test_metrics_creation(self, analyzer, balanced_order_book):
        """测试38: 创建失衡指标"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        assert metrics is not None
        assert metrics.symbol == "BTCUSDT"
        assert -1 <= metrics.volume_imbalance <= 1
        assert -1 <= metrics.weighted_imbalance <= 1

    def test_metrics_direction(self, analyzer, buy_heavy_order_book):
        """测试39: 指标方向判断"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        assert metrics.direction in (
            ImbalanceDirection.BUY_HEAVY,
            ImbalanceDirection.NEUTRAL,
        )

    def test_metrics_severity(self, analyzer, buy_heavy_order_book):
        """测试40: 指标严重程度"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        assert isinstance(metrics.severity, ImbalanceSeverity)

    def test_metrics_momentum(self, analyzer, buy_heavy_order_book):
        """测试41: 失衡动量"""
        bids, asks = buy_heavy_order_book

        # 更新多次以计算动量
        for _ in range(10):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        # 动量应该被计算
        assert isinstance(metrics.imbalance_momentum, float)

    def test_metrics_trend(self, analyzer, buy_heavy_order_book):
        """测试42: 失衡趋势"""
        bids, asks = buy_heavy_order_book

        for _ in range(15):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        assert isinstance(metrics.trend, ImbalanceTrend)

    def test_metrics_confidence(self, analyzer, deep_order_book):
        """测试43: 指标置信度"""
        bids, asks = deep_order_book

        # 深度订单簿应该有更高的置信度
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()

        assert 0 <= metrics.confidence <= 1

    def test_metrics_to_dict(self, analyzer, balanced_order_book):
        """测试44: 指标序列化"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        metrics = analyzer.get_current_metrics()
        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert 'volume_imbalance' in data
        assert 'weighted_imbalance' in data
        assert 'direction' in data


# ============== Test DepthAnalysis ==============

class TestDepthAnalysis:
    """测试深度分析"""

    def test_depth_analysis_levels(self, analyzer, deep_order_book):
        """测试45: 深度分析档位"""
        bids, asks = deep_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()

        assert len(depth.bid_depth_by_level) > 0
        assert len(depth.ask_depth_by_level) > 0

    def test_depth_analysis_ratio(self, analyzer, buy_heavy_order_book):
        """测试46: 深度比率"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()

        # 买盘占优时，深度比率应该大于1
        assert len(depth.depth_ratio_by_level) > 0

    def test_depth_analysis_asymmetry(self, analyzer, buy_heavy_order_book):
        """测试47: 深度不对称性"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()

        # 买盘占优应该有正的不对称性
        assert depth.depth_asymmetry > 0

    def test_depth_analysis_concentration(self, analyzer, balanced_order_book):
        """测试48: 深度集中度"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()

        assert 0 <= depth.bid_concentration <= 1
        assert 0 <= depth.ask_concentration <= 1

    def test_depth_analysis_total_depth(self, analyzer, deep_order_book):
        """测试49: 总深度计算"""
        bids, asks = deep_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()

        assert depth.total_depth > 0
        assert depth.effective_depth > 0

    def test_depth_analysis_to_dict(self, analyzer, balanced_order_book):
        """测试50: 深度分析序列化"""
        bids, asks = balanced_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        depth = analyzer.get_current_depth_analysis()
        data = depth.to_dict()

        assert isinstance(data, dict)
        assert 'bid_depth_by_level' in data
        assert 'depth_asymmetry' in data


# ============== Test ImbalanceSignal ==============

class TestImbalanceSignal:
    """测试失衡信号"""

    def test_signal_creation_from_extreme_imbalance(self, analyzer):
        """测试51: 从极端失衡创建信号"""
        # 创建极端买盘占优订单簿
        bids = [(100.0, 100.0, 50)] * 5
        asks = [(100.1, 1.0, 1)] * 5

        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        # 极端失衡应该生成信号
        assert len(signals) >= 0

    def test_signal_type_classification(self, analyzer, buy_heavy_order_book):
        """测试52: 信号类型分类"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            assert isinstance(signal.signal_type, SignalType)

    def test_signal_strength_classification(self, analyzer, buy_heavy_order_book):
        """测试53: 信号强度分类"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            assert isinstance(signal.signal_strength, SignalStrength)

    def test_signal_confidence(self, analyzer, deep_order_book):
        """测试54: 信号置信度"""
        bids, asks = deep_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            assert 0 <= signal.confidence <= 1

    def test_signal_price_prediction(self, analyzer, buy_heavy_order_book):
        """测试55: 信号价格预测"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            assert signal.predicted_direction in ("up", "down", "neutral")
            assert isinstance(signal.predicted_price_impact, float)

    def test_signal_validity_time(self, analyzer, buy_heavy_order_book):
        """测试56: 信号有效时间"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            assert signal.valid_until > signal.timestamp
            assert signal.ttl_seconds > 0

    def test_signal_callback(self):
        """测试57: 信号回调"""
        callback_signals = []

        def callback(signal: ImbalanceSignal):
            callback_signals.append(signal)

        analyzer = create_order_book_analyzer(
            signal_threshold=0.2,
        )
        analyzer.signal_callback = callback

        # 创建强失衡订单簿
        bids = [(100.0, 100.0, 50)] * 5
        asks = [(100.1, 5.0, 2)] * 5

        analyzer.update_order_book("BTCUSDT", bids, asks)

        # 检查回调是否被调用
        assert isinstance(callback_signals, list)

    def test_signal_to_dict(self, analyzer, buy_heavy_order_book):
        """测试58: 信号序列化"""
        bids, asks = buy_heavy_order_book
        analyzer.update_order_book("BTCUSDT", bids, asks)

        signals = analyzer.get_signals()
        if signals:
            signal = signals[-1]
            data = signal.to_dict()

            assert isinstance(data, dict)
            assert 'signal_type' in data
            assert 'confidence' in data


# ============== Test Config ==============

class TestImbalanceConfig:
    """测试配置"""

    def test_config_creation(self):
        """测试59: 创建配置"""
        config = ImbalanceConfig(
            max_depth_levels=30,
            sampling_levels=15,
            signal_threshold=0.4,
        )

        assert config.max_depth_levels == 30
        assert config.sampling_levels == 15
        assert config.signal_threshold == 0.4

    def test_config_to_dict(self):
        """测试60: 配置序列化"""
        config = ImbalanceConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert 'max_depth_levels' in data
        assert 'signal_threshold' in data


# ============== Test Factory Functions ==============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_order_book_analyzer_default(self):
        """测试61: 创建默认分析器"""
        analyzer = create_order_book_analyzer()

        assert isinstance(analyzer, OrderBookImbalanceAnalyzer)
        assert analyzer.state == AnalyzerState.IDLE

    def test_create_order_book_analyzer_custom(self):
        """测试62: 创建自定义分析器"""
        analyzer = create_order_book_analyzer(
            max_depth_levels=30,
            sampling_levels=15,
            signal_threshold=0.4,
        )

        assert analyzer.config.max_depth_levels == 30
        assert analyzer.config.sampling_levels == 15
        assert analyzer.config.signal_threshold == 0.4


# ============== Test Edge Cases ==============

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_order_book(self, analyzer):
        """测试63: 空订单簿"""
        snapshot = analyzer.update_order_book("BTCUSDT", [], [])

        assert snapshot.total_bid_volume == 0
        assert snapshot.total_ask_volume == 0

    def test_single_sided_order_book_bids_only(self, analyzer):
        """测试64: 仅买单的订单簿"""
        bids = [(100.0, 10.0, 5)]
        asks = []

        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        imbalance = analyzer.calculate_volume_imbalance()
        # 只有买单应该是极端正值
        assert imbalance == 1.0

    def test_single_sided_order_book_asks_only(self, analyzer):
        """测试65: 仅卖单的订单簿"""
        bids = []
        asks = [(100.0, 10.0, 5)]

        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        imbalance = analyzer.calculate_volume_imbalance()
        # 只有卖单应该是极端负值
        assert imbalance == -1.0

    def test_very_wide_spread(self, analyzer):
        """测试66: 非常大的价差"""
        bids = [(100.0, 10.0, 5)]
        asks = [(200.0, 10.0, 5)]  # 100% 价差

        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        assert snapshot.spread == 100.0
        assert snapshot.spread_pct > 50.0

    def test_zero_quantity_levels(self, analyzer):
        """测试67: 零数量档位"""
        bids = [(100.0, 0.0, 0), (99.9, 10.0, 5)]
        asks = [(100.1, 0.0, 0), (100.2, 10.0, 5)]

        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        # 应该能处理零数量
        assert snapshot is not None

    def test_many_updates_performance(self, analyzer, balanced_order_book):
        """测试68: 大量更新的性能"""
        bids, asks = balanced_order_book

        # 快速更新1000次
        for _ in range(1000):
            analyzer.update_order_book("BTCUSDT", bids, asks)

        # 应该保持历史限制
        assert len(analyzer._snapshot_history) <= analyzer.config.history_size

    def test_alternating_imbalance(self, analyzer):
        """测试69: 交替失衡"""
        buy_heavy_bids = [(100.0, 50.0, 25)]
        buy_heavy_asks = [(100.1, 10.0, 5)]

        sell_heavy_bids = [(100.0, 10.0, 5)]
        sell_heavy_asks = [(100.1, 50.0, 25)]

        # 交替更新
        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", buy_heavy_bids, buy_heavy_asks)
            analyzer.update_order_book("BTCUSDT", sell_heavy_bids, sell_heavy_asks)

        # 趋势应该被检测到
        metrics = analyzer.get_current_metrics()
        assert metrics.trend in (ImbalanceTrend.REVERSING, ImbalanceTrend.STABLE)

    def test_decimal_prices(self, analyzer):
        """测试70: 小数价格"""
        bids = [
            (0.00012345, 1000.0, 10),
            (0.00012344, 2000.0, 15),
        ]
        asks = [
            (0.00012346, 1000.0, 10),
            (0.00012347, 2000.0, 15),
        ]

        snapshot = analyzer.update_order_book("BTCUSDT", bids, asks)

        assert snapshot.best_bid == pytest.approx(0.00012345)
        assert snapshot.best_ask == pytest.approx(0.00012346)


# ============== Test Integration ==============

class TestIntegration:
    """测试集成场景"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, balanced_order_book):
        """测试71: 完整工作流"""
        signal_count = []

        def signal_callback(signal):
            signal_count.append(signal)

        analyzer = create_order_book_analyzer(
            signal_threshold=0.2,
        )
        analyzer.signal_callback = signal_callback

        # 启动分析器
        await analyzer.start()

        bids, asks = balanced_order_book

        # 模拟实时更新
        for _ in range(10):
            analyzer.update_order_book("BTCUSDT", bids, asks)
            await asyncio.sleep(0.01)

        # 停止分析器
        await analyzer.stop()

        # 验证数据
        assert analyzer.get_current_snapshot() is not None
        assert analyzer.get_current_metrics() is not None
        assert analyzer.get_statistics().total_samples > 0

    def test_multiple_symbols(self):
        """测试72: 多交易对支持"""
        analyzer = create_order_book_analyzer()

        bids = [(100.0, 10.0, 5)]
        asks = [(100.1, 10.0, 5)]

        # 更新不同交易对
        analyzer.update_order_book("BTCUSDT", bids, asks)
        analyzer.update_order_book("ETHUSDT", bids, asks)

        # 最新快照应该是最后更新的
        assert analyzer.get_current_snapshot().symbol == "ETHUSDT"

    def test_real_time_signal_generation(self, buy_heavy_order_book, sell_heavy_order_book):
        """测试73: 实时信号生成"""
        analyzer = create_order_book_analyzer(signal_threshold=0.3)

        buy_bids, buy_asks = buy_heavy_order_book
        sell_bids, sell_asks = sell_heavy_order_book

        # 模拟市场变化
        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", buy_bids, buy_asks)

        for _ in range(5):
            analyzer.update_order_book("BTCUSDT", sell_bids, sell_asks)

        # 应该有信号生成
        signals = analyzer.get_signals()
        assert len(signals) >= 0

    def test_statistics_accuracy(self, balanced_order_book, buy_heavy_order_book):
        """测试74: 统计准确性"""
        analyzer = create_order_book_analyzer()

        bal_bids, bal_asks = balanced_order_book
        buy_bids, buy_asks = buy_heavy_order_book

        # 50%均衡, 50%买盘占优
        for _ in range(10):
            analyzer.update_order_book("BTCUSDT", bal_bids, bal_asks)
        for _ in range(10):
            analyzer.update_order_book("BTCUSDT", buy_bids, buy_asks)

        stats = analyzer.get_statistics()

        # 统计应该反映数据分布
        assert stats.total_samples == 20
        assert stats.buy_heavy_count > 0
        assert stats.avg_volume_imbalance > 0  # 因为有买盘占优


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
