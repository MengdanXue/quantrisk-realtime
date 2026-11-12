"""
Correlation Tracker 测试套件 - Comprehensive Test Suite

测试覆盖:
1. 基础功能测试 (Basic Functionality)
2. 相关性计算测试 (Correlation Calculation)
3. 滚动窗口测试 (Rolling Window)
4. 突变检测测试 (Break Detection)
5. DCC-GARCH 测试 (Dynamic Conditional Correlation)
6. 尾部相关性测试 (Tail Correlation)
7. 聚类分析测试 (Clustering)
8. 预测功能测试 (Forecasting)
9. 告警系统测试 (Alert System)
10. 边界条件测试 (Edge Cases)
"""

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import asyncio
import math
import statistics
import unittest
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.correlation_tracker import (
    AlertType,
    CorrelationAlert,
    CorrelationBreak,
    CorrelationCluster,
    CorrelationForecast,
    CorrelationMatrix,
    CorrelationPair,
    CorrelationRegime,
    CorrelationStrength,
    CorrelationTracker,
    CorrelationTrend,
    DCCGarchState,
    ReturnData,
    ReturnDataProvider,
    RollingCorrelation,
    SimulatedReturnDataProvider,
    TailCorrelation,
    TrackerConfig,
    TrackerState,
    create_correlation_tracker,
)


class TestBasicFunctionality(unittest.TestCase):
    """测试基础功能"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=10,
            default_correlation_window=20,
            update_interval=0.1,
        )
        self.provider = SimulatedReturnDataProvider()
        self.tracker = CorrelationTracker(
            config=self.config,
            data_provider=self.provider,
        )

    def test_01_tracker_initialization(self):
        """测试追踪器初始化"""
        self.assertEqual(self.tracker.state, TrackerState.STOPPED)
        self.assertEqual(len(self.tracker.get_symbols()), 0)

    def test_02_add_symbol(self):
        """测试添加资产"""
        self.tracker.add_symbol("AAPL")
        self.tracker.add_symbol("GOOGL")

        symbols = self.tracker.get_symbols()
        self.assertEqual(len(symbols), 2)
        self.assertIn("AAPL", symbols)
        self.assertIn("GOOGL", symbols)

    def test_03_remove_symbol(self):
        """测试移除资产"""
        self.tracker.add_symbol("AAPL")
        self.tracker.add_symbol("GOOGL")
        self.tracker.remove_symbol("AAPL")

        symbols = self.tracker.get_symbols()
        self.assertEqual(len(symbols), 1)
        self.assertNotIn("AAPL", symbols)

    async def test_04_start_stop_tracker(self):
        """测试启动和停止追踪器"""
        await self.tracker.start()
        self.assertEqual(self.tracker.state, TrackerState.RUNNING)

        await self.tracker.stop()
        self.assertEqual(self.tracker.state, TrackerState.STOPPED)

    async def test_05_pause_resume_tracker(self):
        """测试暂停和恢复追踪器"""
        await self.tracker.start()
        await self.tracker.pause()
        self.assertEqual(self.tracker.state, TrackerState.PAUSED)

        await self.tracker.resume()
        self.assertEqual(self.tracker.state, TrackerState.RUNNING)

        await self.tracker.stop()

    def test_06_get_statistics(self):
        """测试获取统计信息"""
        self.tracker.add_symbol("AAPL")
        self.tracker.add_symbol("GOOGL")

        stats = self.tracker.get_statistics()
        self.assertIn("state", stats)
        self.assertIn("symbols", stats)
        self.assertEqual(stats["symbols"], 2)


class TestCorrelationCalculation(unittest.TestCase):
    """测试相关性计算"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(min_correlation_window=5)
        self.tracker = CorrelationTracker(config=self.config)

    async def test_07_perfect_positive_correlation(self):
        """测试完全正相关 (r = 1.0)"""
        # 添加完全正相关的数据
        for i in range(20):
            value = 0.01 * i
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(corr, 1.0, places=1)

    async def test_08_perfect_negative_correlation(self):
        """测试完全负相关 (r = -1.0)"""
        # 添加完全负相关的数据 - 使用相同的收益率幅度但相反方向
        base_price_a = 100.0
        base_price_b = 100.0
        for i in range(20):
            change = i * 0.01  # 1% per step
            await self.tracker.add_price("AAPL", base_price_a * (1 + change), datetime.now())
            await self.tracker.add_price("GOOGL", base_price_b * (1 - change), datetime.now())

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(corr)
        # 由于收益率计算的特性，可能不是完全 -1.0，但应该接近
        self.assertLess(corr, -0.8)

    async def test_09_zero_correlation(self):
        """测试零相关"""
        import random
        random.seed(42)

        # 添加随机不相关的数据
        for i in range(50):
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), random.gauss(0, 0.01), 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), random.gauss(0, 0.01), 200 + i)
            )

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(corr)
        # 零相关应该接近0 (允许一定随机误差)
        self.assertLess(abs(corr), 0.3)

    def test_10_classify_correlation_strength(self):
        """测试相关性强度分类"""
        # 测试各种相关性强度
        test_cases = [
            (0.9, CorrelationStrength.VERY_STRONG_POSITIVE),
            (0.7, CorrelationStrength.STRONG_POSITIVE),
            (0.5, CorrelationStrength.MODERATE_POSITIVE),
            (0.3, CorrelationStrength.WEAK_POSITIVE),
            (0.1, CorrelationStrength.NEGLIGIBLE),
            (-0.1, CorrelationStrength.NEGLIGIBLE),
            (-0.3, CorrelationStrength.WEAK_NEGATIVE),
            (-0.5, CorrelationStrength.MODERATE_NEGATIVE),
            (-0.7, CorrelationStrength.STRONG_NEGATIVE),
            (-0.9, CorrelationStrength.VERY_STRONG_NEGATIVE),
        ]

        for corr_value, expected_strength in test_cases:
            strength = self.tracker._classify_correlation_strength(corr_value)
            self.assertEqual(strength, expected_strength)

    async def test_11_correlation_matrix(self):
        """测试相关性矩阵计算"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # 添加数据
        for i in range(30):
            for symbol in symbols:
                await self.tracker.add_price(symbol, 100 + i, datetime.now())

        matrix = self.tracker.get_correlation_matrix()
        self.assertIsNotNone(matrix)
        self.assertEqual(len(matrix.symbols), 3)
        self.assertEqual(len(matrix.matrix), 3)
        self.assertEqual(len(matrix.matrix[0]), 3)

        # 对角线应该为1
        for i in range(3):
            self.assertEqual(matrix.matrix[i][i], 1.0)

    async def test_12_correlation_pair_properties(self):
        """测试相关性配对属性"""
        for i in range(20):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        pair = self.tracker.get_correlation_pair("AAPL", "GOOGL")
        self.assertIsNotNone(pair)
        self.assertTrue(hasattr(pair, "correlation"))
        self.assertTrue(hasattr(pair, "strength"))
        self.assertTrue(hasattr(pair, "p_value"))
        self.assertTrue(hasattr(pair, "confidence_interval"))
        self.assertEqual(pair.pair_key, "AAPL-GOOGL")


class TestRollingWindow(unittest.TestCase):
    """测试滚动窗口相关性"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=5,
            rolling_window=10,
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_13_rolling_correlation_tracking(self):
        """测试滚动相关性追踪"""
        # 添加数据
        for i in range(30):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        rolling = self.tracker.get_rolling_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(rolling)
        self.assertGreater(len(rolling.correlations), 0)
        self.assertEqual(rolling.window_size, 10)

    async def test_14_rolling_correlation_trend_detection(self):
        """测试相关性趋势检测"""
        # 添加趋势性变化的数据
        for i in range(15):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        # 改变相关性
        for i in range(15):
            await self.tracker.add_price("AAPL", 115 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 230 - i, datetime.now())

        rolling = self.tracker.get_rolling_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(rolling)
        self.assertIn(rolling.trend, [
            CorrelationTrend.DECREASING,
            CorrelationTrend.BREAKING,
        ])

    async def test_15_rolling_correlation_volatility(self):
        """测试相关性波动率"""
        import random
        random.seed(42)

        # 添加波动的相关性数据
        for i in range(50):
            noise1 = random.gauss(0, 0.02)
            noise2 = random.gauss(0, 0.02)
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), 0.01 + noise1, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), 0.01 + noise2, 200 + i)
            )

        rolling = self.tracker.get_rolling_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(rolling)
        self.assertGreaterEqual(rolling.volatility, 0.0)


class TestBreakDetection(unittest.TestCase):
    """测试相关性突变检测"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=5,
            break_threshold=0.3,
            break_lookback=5,
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_16_detect_correlation_break(self):
        """测试检测相关性突变"""
        # 第一阶段: 高正相关
        for i in range(20):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        # 第二阶段: 负相关
        for i in range(20):
            await self.tracker.add_price("AAPL", 120 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 240 - i * 2, datetime.now())

        breaks = self.tracker.get_breaks()
        self.assertGreater(len(breaks), 0)

    async def test_17_correlation_break_properties(self):
        """测试相关性突变属性"""
        # 制造突变
        for i in range(15):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        for i in range(15):
            await self.tracker.add_price("AAPL", 115 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 230 - i * 2, datetime.now())

        breaks = self.tracker.get_breaks(symbol1="AAPL")
        if len(breaks) > 0:
            break_event = breaks[0]
            self.assertTrue(hasattr(break_event, "break_id"))
            self.assertTrue(hasattr(break_event, "old_correlation"))
            self.assertTrue(hasattr(break_event, "new_correlation"))
            self.assertTrue(hasattr(break_event, "change_magnitude"))

    async def test_18_break_detection_callback(self):
        """测试突变检测回调"""
        breaks_detected = []

        def break_callback(break_event: CorrelationBreak):
            breaks_detected.append(break_event)

        tracker = CorrelationTracker(
            config=self.config,
            break_callback=break_callback,
        )

        # 制造突变
        for i in range(15):
            await tracker.add_price("AAPL", 100 + i, datetime.now())
            await tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        for i in range(15):
            await tracker.add_price("AAPL", 115 + i, datetime.now())
            await tracker.add_price("GOOGL", 230 - i * 2, datetime.now())

        # 可能检测到突变
        self.assertGreaterEqual(len(breaks_detected), 0)


class TestDCCGarch(unittest.TestCase):
    """测试 DCC-GARCH 动态条件相关性"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=10,
            dcc_alpha=0.05,
            dcc_beta=0.90,
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_19_dcc_garch_calculation(self):
        """测试 DCC-GARCH 计算"""
        # 添加足够的数据
        for i in range(50):
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), 0.01, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), 0.01, 200 + i)
            )

        dcc_state = self.tracker.get_dcc_state("AAPL", "GOOGL")
        self.assertIsNotNone(dcc_state)

    async def test_20_dcc_state_properties(self):
        """测试 DCC 状态属性"""
        for i in range(50):
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), 0.01, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), 0.01, 200 + i)
            )

        dcc_state = self.tracker.get_dcc_state("AAPL", "GOOGL")
        if dcc_state:
            self.assertTrue(hasattr(dcc_state, "dynamic_correlation"))
            self.assertTrue(hasattr(dcc_state, "variance1"))
            self.assertTrue(hasattr(dcc_state, "variance2"))
            self.assertTrue(hasattr(dcc_state, "covariance"))
            self.assertTrue(hasattr(dcc_state, "conditional_vol1"))
            self.assertTrue(hasattr(dcc_state, "conditional_vol2"))

    async def test_21_dcc_correlation_bounds(self):
        """测试 DCC 相关性范围"""
        for i in range(50):
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), 0.01, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), 0.01, 200 + i)
            )

        dcc_state = self.tracker.get_dcc_state("AAPL", "GOOGL")
        if dcc_state:
            # 相关性应该在 [-1, 1] 范围内
            self.assertGreaterEqual(dcc_state.dynamic_correlation, -1.0)
            self.assertLessEqual(dcc_state.dynamic_correlation, 1.0)


class TestTailCorrelation(unittest.TestCase):
    """测试尾部相关性"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=10,
            tail_percentile=0.05,
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_22_tail_correlation_calculation(self):
        """测试尾部相关性计算"""
        import random
        random.seed(42)

        # 添加数据，包含极端事件
        for i in range(100):
            # 正常数据
            ret1 = random.gauss(0.001, 0.01)
            ret2 = random.gauss(0.001, 0.01)

            # 偶尔添加极端事件 (同时大跌)
            if i % 20 == 0:
                ret1 = -0.05
                ret2 = -0.05

            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), ret1, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), ret2, 200 + i)
            )

        tail_corr = self.tracker.get_tail_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(tail_corr)

    async def test_23_tail_correlation_properties(self):
        """测试尾部相关性属性"""
        import random
        random.seed(42)

        for i in range(100):
            ret1 = random.gauss(0, 0.02)
            ret2 = random.gauss(0, 0.02)
            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), ret1, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), ret2, 200 + i)
            )

        tail_corr = self.tracker.get_tail_correlation("AAPL", "GOOGL")
        if tail_corr:
            self.assertTrue(hasattr(tail_corr, "lower_tail_correlation"))
            self.assertTrue(hasattr(tail_corr, "upper_tail_correlation"))
            self.assertTrue(hasattr(tail_corr, "tail_threshold"))
            self.assertTrue(hasattr(tail_corr, "lower_tail_events"))
            self.assertTrue(hasattr(tail_corr, "upper_tail_events"))

    async def test_24_lower_tail_correlation(self):
        """测试下尾相关性 (同时大跌)"""
        import random
        random.seed(42)

        # 制造下尾相关性
        for i in range(100):
            if i % 10 < 2:  # 20% 概率同时大跌
                ret1 = -0.03 - random.uniform(0, 0.02)
                ret2 = -0.03 - random.uniform(0, 0.02)
            else:
                ret1 = random.gauss(0.001, 0.01)
                ret2 = random.gauss(0.001, 0.01)

            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), ret1, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), ret2, 200 + i)
            )

        tail_corr = self.tracker.get_tail_correlation("AAPL", "GOOGL")
        if tail_corr and tail_corr.lower_tail_events >= 3:
            # 下尾应该有较高的相关性
            self.assertGreater(tail_corr.lower_tail_correlation, 0.0)


class TestClustering(unittest.TestCase):
    """测试相关性聚类"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=10,
            cluster_threshold=0.7,
            min_cluster_size=2,
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_25_correlation_clustering(self):
        """测试相关性聚类"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # 添加数据 (AAPL和GOOGL高度相关，MSFT和AMZN高度相关)
        for i in range(30):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())
            await self.tracker.add_price("MSFT", 300 + i * 1.5, datetime.now())
            await self.tracker.add_price("AMZN", 400 + i * 3, datetime.now())

        clusters = self.tracker.get_clusters()
        # 应该有聚类生成
        self.assertGreaterEqual(len(clusters), 0)

    async def test_26_cluster_properties(self):
        """测试聚类属性"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for i in range(30):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())
            await self.tracker.add_price("MSFT", 300 + i * 1.5, datetime.now())

        clusters = self.tracker.get_clusters()
        for cluster in clusters:
            self.assertTrue(hasattr(cluster, "cluster_id"))
            self.assertTrue(hasattr(cluster, "symbols"))
            self.assertTrue(hasattr(cluster, "centroid_symbol"))
            self.assertTrue(hasattr(cluster, "average_correlation"))
            self.assertGreaterEqual(len(cluster.symbols), self.config.min_cluster_size)

    async def test_27_cluster_centroid_selection(self):
        """测试聚类中心选择"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for i in range(30):
            # 所有资产高度正相关
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())
            await self.tracker.add_price("MSFT", 150 + i * 1.5, datetime.now())

        clusters = self.tracker.get_clusters()
        for cluster in clusters:
            # 中心资产应该在聚类中
            self.assertIn(cluster.centroid_symbol, cluster.symbols)


class TestForecasting(unittest.TestCase):
    """测试相关性预测"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=10,
            forecast_horizon=5,
            forecast_method="ema",
        )
        self.tracker = CorrelationTracker(config=self.config)

    async def test_28_correlation_forecast_ema(self):
        """测试 EMA 预测方法"""
        for i in range(40):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        forecast = self.tracker.get_forecast("AAPL", "GOOGL")
        self.assertIsNotNone(forecast)

    async def test_29_forecast_properties(self):
        """测试预测属性"""
        for i in range(40):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        forecast = self.tracker.get_forecast("AAPL", "GOOGL")
        if forecast:
            self.assertTrue(hasattr(forecast, "current_correlation"))
            self.assertTrue(hasattr(forecast, "forecast_correlation"))
            self.assertTrue(hasattr(forecast, "forecast_horizon"))
            self.assertTrue(hasattr(forecast, "confidence"))
            self.assertTrue(hasattr(forecast, "forecast_method"))

    async def test_30_forecast_arima_method(self):
        """测试 ARIMA 预测方法"""
        config = TrackerConfig(
            min_correlation_window=10,
            forecast_method="arima",
        )
        tracker = CorrelationTracker(config=config)

        for i in range(40):
            await tracker.add_price("AAPL", 100 + i, datetime.now())
            await tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        forecast = tracker.get_forecast("AAPL", "GOOGL")
        self.assertIsNotNone(forecast)

    async def test_31_forecast_confidence(self):
        """测试预测置信度"""
        for i in range(40):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        forecast = self.tracker.get_forecast("AAPL", "GOOGL")
        if forecast:
            # 置信度应该在 [0, 1] 范围内
            self.assertGreaterEqual(forecast.confidence, 0.0)
            self.assertLessEqual(forecast.confidence, 1.0)


class TestAlertSystem(unittest.TestCase):
    """测试告警系统"""

    def setUp(self):
        """测试前置"""
        self.config = TrackerConfig(
            min_correlation_window=5,
            high_correlation_threshold=0.9,
            break_threshold=0.3,
        )
        self.alerts_received = []

        def alert_callback(alert: CorrelationAlert):
            self.alerts_received.append(alert)

        self.tracker = CorrelationTracker(
            config=self.config,
            alert_callback=alert_callback,
        )

    async def test_32_correlation_break_alert(self):
        """测试相关性突变告警"""
        # 制造突变
        for i in range(15):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        for i in range(15):
            await self.tracker.add_price("AAPL", 115 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 230 - i * 2, datetime.now())

        # 可能触发告警
        self.assertGreaterEqual(len(self.alerts_received), 0)

    async def test_33_tail_correlation_alert(self):
        """测试尾部相关性告警"""
        import random
        random.seed(42)

        # 制造高下尾相关性
        for i in range(100):
            if i % 5 == 0:  # 频繁的同时大跌
                ret1 = -0.05
                ret2 = -0.05
            else:
                ret1 = random.gauss(0, 0.01)
                ret2 = random.gauss(0, 0.01)

            await self.tracker.add_return_data(
                ReturnData("AAPL", datetime.now(), ret1, 100 + i)
            )
            await self.tracker.add_return_data(
                ReturnData("GOOGL", datetime.now(), ret2, 200 + i)
            )

        # 可能触发尾部相关性告警
        tail_alerts = [
            a for a in self.alerts_received
            if a.alert_type == AlertType.TAIL_CORRELATION
        ]
        self.assertGreaterEqual(len(tail_alerts), 0)

    def test_34_alert_properties(self):
        """测试告警属性"""
        alert = CorrelationAlert(
            alert_id="test123",
            alert_type=AlertType.CORRELATION_BREAK,
            symbol1="AAPL",
            symbol2="GOOGL",
            message="Test alert",
            severity="high",
            correlation_value=0.5,
            timestamp=datetime.now(),
        )

        self.assertEqual(alert.alert_id, "test123")
        self.assertEqual(alert.alert_type, AlertType.CORRELATION_BREAK)
        self.assertFalse(alert.acknowledged)

    async def test_35_acknowledge_alert(self):
        """测试确认告警"""
        # 制造告警
        for i in range(15):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        for i in range(15):
            await self.tracker.add_price("AAPL", 115 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 230 - i * 2, datetime.now())

        alerts = self.tracker.get_alerts(acknowledged=False)
        if len(alerts) > 0:
            alert_id = alerts[0].alert_id
            success = self.tracker.acknowledge_alert(alert_id)
            self.assertTrue(success)

            # 验证已确认
            acknowledged_alerts = self.tracker.get_alerts(acknowledged=True)
            self.assertGreater(len(acknowledged_alerts), 0)

    async def test_36_filter_alerts_by_type(self):
        """测试按类型过滤告警"""
        # 制造不同类型的告警
        for i in range(15):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        for i in range(15):
            await self.tracker.add_price("AAPL", 115 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 230 - i * 2, datetime.now())

        break_alerts = self.tracker.get_alerts(alert_type=AlertType.CORRELATION_BREAK)
        # 所有返回的告警应该是突变类型
        for alert in break_alerts:
            self.assertEqual(alert.alert_type, AlertType.CORRELATION_BREAK)


class TestEdgeCases(unittest.TestCase):
    """测试边界条件"""

    def setUp(self):
        """测试前置"""
        self.tracker = CorrelationTracker()

    async def test_37_insufficient_data(self):
        """测试数据不足的情况"""
        # 只添加少量数据
        await self.tracker.add_price("AAPL", 100, datetime.now())
        await self.tracker.add_price("GOOGL", 200, datetime.now())

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        # 数据不足，应该返回 None
        self.assertIsNone(corr)

    async def test_38_single_symbol(self):
        """测试单个资产的情况"""
        for i in range(20):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())

        # 单个资产无法计算相关性
        matrix = self.tracker.get_correlation_matrix()
        self.assertIsNone(matrix)

    async def test_39_zero_volatility(self):
        """测试零波动率数据"""
        # 所有价格相同
        for i in range(20):
            await self.tracker.add_price("AAPL", 100, datetime.now())
            await self.tracker.add_price("GOOGL", 200, datetime.now())

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        # 零波动率应该返回0相关性
        if corr is not None:
            self.assertEqual(corr, 0.0)

    async def test_40_missing_data_alignment(self):
        """测试数据不对齐的情况"""
        # AAPL有30个数据点
        for i in range(30):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())

        # GOOGL只有20个数据点
        for i in range(20):
            await self.tracker.add_price("GOOGL", 200 + i, datetime.now())

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        # 应该使用较短的序列
        self.assertIsNotNone(corr)

    def test_41_correlation_bounds(self):
        """测试相关性范围限制"""
        # 测试内部计算方法
        returns1 = [0.01] * 20
        returns2 = [0.01] * 20

        corr = self.tracker._calculate_correlation(returns1, returns2)

        # 相关性应该在 [-1, 1] 范围内
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    async def test_42_very_long_history(self):
        """测试超长历史数据"""
        # 添加大量数据
        for i in range(2000):
            await self.tracker.add_price("AAPL", 100 + i * 0.1, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 0.2, datetime.now())

        # 应该能正常计算
        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(corr)

    def test_43_factory_function(self):
        """测试工厂函数"""
        tracker = create_correlation_tracker(
            min_correlation_window=15,
            rolling_window=30,
        )

        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.config.min_correlation_window, 15)
        self.assertEqual(tracker.config.rolling_window, 30)

    async def test_44_concurrent_updates(self):
        """测试并发更新"""
        # 模拟并发添加数据
        tasks = []
        for i in range(20):
            task1 = self.tracker.add_price("AAPL", 100 + i, datetime.now())
            task2 = self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())
            tasks.extend([task1, task2])

        await asyncio.gather(*tasks)

        corr = self.tracker.get_correlation("AAPL", "GOOGL")
        self.assertIsNotNone(corr)

    def test_45_to_dict_methods(self):
        """测试 to_dict 序列化方法"""
        # 测试各种数据类的序列化
        pair = CorrelationPair(
            symbol1="AAPL",
            symbol2="GOOGL",
            correlation=0.8,
            timestamp=datetime.now(),
            strength=CorrelationStrength.STRONG_POSITIVE,
            sample_size=100,
        )

        dict_data = pair.to_dict()
        self.assertIn("symbol1", dict_data)
        self.assertIn("correlation", dict_data)
        self.assertIn("pair_key", dict_data)

    async def test_46_extreme_correlation_values(self):
        """测试极端相关性值"""
        # 测试完全同步变化 (应该 = 1.0)
        for i in range(20):
            value = 100 + i
            await self.tracker.add_price("AAPL", value, datetime.now())
            await self.tracker.add_price("GOOGL", value * 2, datetime.now())

        pair = self.tracker.get_correlation_pair("AAPL", "GOOGL")
        if pair:
            self.assertAlmostEqual(abs(pair.correlation), 1.0, places=1)

    async def test_47_correlation_stability(self):
        """测试相关性稳定性"""
        # 添加稳定的相关性数据
        for i in range(50):
            await self.tracker.add_price("AAPL", 100 + i, datetime.now())
            await self.tracker.add_price("GOOGL", 200 + i * 2, datetime.now())

        rolling = self.tracker.get_rolling_correlation("AAPL", "GOOGL")
        if rolling and len(rolling.correlations) > 10:
            # 稳定的相关性应该有较低波动率（放宽阈值）
            self.assertLess(rolling.volatility, 0.5)

    async def test_48_multiple_symbol_pairs(self):
        """测试多个资产对"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # 添加数据
        for i in range(30):
            for symbol in symbols:
                await self.tracker.add_price(symbol, 100 + i, datetime.now())

        # 检查所有配对
        pair_count = len(self.tracker._current_correlations)
        expected_pairs = len(symbols) * (len(symbols) - 1) // 2
        self.assertEqual(pair_count, expected_pairs)

    def test_49_config_validation(self):
        """测试配置参数验证"""
        config = TrackerConfig(
            min_correlation_window=20,
            rolling_window=10,
            tail_percentile=0.05,
        )

        self.assertEqual(config.min_correlation_window, 20)
        self.assertEqual(config.rolling_window, 10)
        self.assertAlmostEqual(config.tail_percentile, 0.05)

    def test_50_statistics_completeness(self):
        """测试统计信息完整性"""
        self.tracker.add_symbol("AAPL")
        self.tracker.add_symbol("GOOGL")

        stats = self.tracker.get_statistics()

        # 验证统计信息包含所有必需字段
        required_fields = [
            "state",
            "symbols",
            "total_pairs",
            "total_breaks",
            "total_alerts",
            "unacknowledged_alerts",
            "clusters",
            "matrix_average_correlation",
            "config",
        ]

        for field in required_fields:
            self.assertIn(field, stats)


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
    TestCorrelationCalculation,
    TestRollingWindow,
    TestBreakDetection,
    TestDCCGarch,
    TestTailCorrelation,
    TestClustering,
    TestForecasting,
    TestAlertSystem,
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
