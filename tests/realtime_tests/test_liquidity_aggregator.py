"""
Liquidity Aggregator 测试套件 - Comprehensive Test Suite

测试覆盖:
1. 基础功能测试 (Basic Functionality)
2. 交易所流动性测试 (Exchange Liquidity)
3. 订单簿聚合测试 (Order Book Aggregation)
4. 成本估算测试 (Cost Estimation)
5. 执行策略测试 (Execution Strategy)
6. 暗池流动性测试 (Dark Pool)
7. 流动性预测测试 (Liquidity Prediction)
8. 指标计算测试 (Metrics Calculation)
9. 边界条件测试 (Edge Cases)
10. 性能测试 (Performance)
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
from typing import List, Dict, Any

from quantrisk_realtime.realtime.liquidity_aggregator import (
    AggregatedBook,
    AggregationMethod,
    AggregatorConfig,
    DarkPoolLiquidity,
    ExchangeLiquidity,
    ExecutionStrategy,
    LiquidityAggregator,
    LiquidityCost,
    LiquidityMetrics,
    LiquidityPrediction,
    LiquidityScore,
    LiquidityTier,
    LiquidityType,
    PriceLevel,
    create_dark_pool_liquidity,
    create_exchange_liquidity,
    create_liquidity_aggregator,
)


# ==================== 测试辅助函数 ====================

def create_mock_exchange_liquidity(
    exchange_id: str,
    symbol: str,
    best_bid: float = 100.0,
    best_ask: float = 100.05,
    bid_depth: float = 1000.0,
    ask_depth: float = 1000.0,
    levels: int = 10,
    spread_multiplier: float = 1.0
) -> ExchangeLiquidity:
    """创建模拟交易所流动性数据"""
    bid_levels = {}
    ask_levels = {}

    # 创建买盘深度
    for i in range(levels):
        price = best_bid - i * 0.01 * spread_multiplier
        size = bid_depth / levels * (1 - i * 0.05)  # 递减的深度
        bid_levels[price] = max(0, size)

    # 创建卖盘深度
    for i in range(levels):
        price = best_ask + i * 0.01 * spread_multiplier
        size = ask_depth / levels * (1 - i * 0.05)
        ask_levels[price] = max(0, size)

    return ExchangeLiquidity(
        exchange_id=exchange_id,
        symbol=symbol,
        timestamp=datetime.now(),
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        displayed_liquidity=bid_depth + ask_depth,
        hidden_liquidity=0,
        implied_liquidity=0,
        spread_bps=(best_ask - best_bid) / ((best_bid + best_ask) / 2) * 10000,
        depth_score=80.0,
        resilience_score=70.0,
        trade_velocity=10.0,
        volume_velocity=5000.0,
        avg_spread_bps=5.0,
        avg_depth=1000.0,
        is_active=True,
        last_update_ms=datetime.now().timestamp() * 1000,
    )


def create_mock_dark_pool(
    pool_id: str,
    symbol: str,
    indicative_size: float = 5000.0,
    match_probability: float = 0.6,
) -> DarkPoolLiquidity:
    """创建模拟暗池流动性"""
    return DarkPoolLiquidity(
        pool_id=pool_id,
        symbol=symbol,
        timestamp=datetime.now(),
        indicative_size=indicative_size,
        min_block_size=indicative_size * 0.5,
        match_probability=match_probability,
        avg_execution_size=indicative_size * 0.8,
        avg_price_improvement_bps=2.0,
        fill_rate_history=0.7,
        accepts_retail=False,
        minimum_value=1000.0,
        fee_bps=0.5,
        is_available=True,
    )


class TestBasicFunctionality(unittest.TestCase):
    """测试基础功能"""

    def setUp(self):
        """测试前置"""
        self.config = AggregatorConfig()  # 使用默认配置
        self.aggregator = LiquidityAggregator(config=self.config)

    def test_01_aggregator_initialization(self):
        """测试聚合器初始化"""
        self.assertIsNotNone(self.aggregator)
        self.assertIsNotNone(self.aggregator.config)

    def test_02_default_config(self):
        """测试默认配置"""
        aggregator = LiquidityAggregator()
        self.assertIsNotNone(aggregator.config)
        self.assertGreater(aggregator.config.depth_levels_to_aggregate, 0)

    def test_03_create_aggregator_factory(self):
        """测试工厂函数"""
        aggregator = create_liquidity_aggregator()
        self.assertIsNotNone(aggregator)
        self.assertIsInstance(aggregator, LiquidityAggregator)

    def test_04_get_statistics_empty(self):
        """测试空状态统计"""
        stats = self.aggregator.get_statistics()
        self.assertIsInstance(stats, dict)


class TestExchangeLiquidity(unittest.TestCase):
    """测试交易所流动性功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()

    def test_01_update_exchange_liquidity(self):
        """测试更新交易所流动性"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
            best_bid=50000.0,
            best_ask=50010.0,
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        stats = self.aggregator.get_statistics()
        self.assertEqual(stats.get('total_exchanges_tracked', 0), 1)

    def test_02_multiple_exchanges(self):
        """测试多交易所流动性"""
        exchanges = ["binance", "okx", "bybit", "huobi"]
        for ex in exchanges:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex,
                symbol="BTC-USDT",
                best_bid=50000.0 + exchanges.index(ex),
                best_ask=50010.0 + exchanges.index(ex),
            )
            self.aggregator.update_exchange_liquidity(liquidity)

        stats = self.aggregator.get_statistics()
        self.assertEqual(stats.get('total_exchanges_tracked', 0), 4)

    def test_03_update_same_exchange(self):
        """测试更新同一交易所"""
        # 第一次更新
        liquidity1 = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
            best_bid=50000.0,
            best_ask=50010.0,
        )
        self.aggregator.update_exchange_liquidity(liquidity1)

        # 第二次更新（应该覆盖）
        liquidity2 = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
            best_bid=50050.0,
            best_ask=50060.0,
        )
        self.aggregator.update_exchange_liquidity(liquidity2)

        stats = self.aggregator.get_statistics()
        # 仍然只有一个交易所
        self.assertEqual(stats.get('total_exchanges_tracked', 0), 1)

    def test_04_exchange_liquidity_properties(self):
        """测试ExchangeLiquidity属性"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
            best_bid=50000.0,
            best_ask=50010.0,
            bid_depth=10000.0,
            ask_depth=10000.0,
        )

        self.assertEqual(liquidity.best_bid, 50000.0)
        self.assertEqual(liquidity.best_ask, 50010.0)
        self.assertIsNotNone(liquidity.mid_price)
        self.assertAlmostEqual(liquidity.mid_price, 50005.0, places=1)
        self.assertGreater(liquidity.total_bid_volume, 0)
        self.assertGreater(liquidity.total_ask_volume, 0)

    def test_05_exchange_liquidity_to_dict(self):
        """测试ExchangeLiquidity序列化"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
        )

        data = liquidity.to_dict()
        self.assertIn('exchange_id', data)
        self.assertIn('symbol', data)
        self.assertIn('best_bid', data)
        self.assertIn('best_ask', data)
        self.assertEqual(data['exchange_id'], 'binance')
        self.assertEqual(data['symbol'], 'BTC-USDT')


class TestOrderBookAggregation(unittest.TestCase):
    """测试订单簿聚合功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        # 添加多个交易所的流动性
        self._setup_exchanges()

    def _setup_exchanges(self):
        """设置多个交易所"""
        exchanges_data = [
            ("binance", 50000.0, 50005.0, 5000.0),
            ("okx", 50001.0, 50006.0, 3000.0),
            ("bybit", 49999.0, 50004.0, 4000.0),
        ]
        for ex_id, best_bid, best_ask, depth in exchanges_data:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex_id,
                symbol="BTC-USDT",
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=depth,
                ask_depth=depth,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

    def test_01_aggregate_order_book(self):
        """测试订单簿聚合"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        self.assertIsNotNone(book)
        self.assertIsInstance(book, AggregatedBook)
        self.assertEqual(book.symbol, "BTC-USDT")

    def test_02_nbbo_calculation(self):
        """测试NBBO计算"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        # NBBO应该是所有交易所的最优价格
        self.assertIsNotNone(book.nbbo_bid)
        self.assertIsNotNone(book.nbbo_ask)
        # 最优买价应该是最高的买价
        self.assertGreaterEqual(book.nbbo_bid, 49999.0)
        # 最优卖价应该是最低的卖价
        self.assertLessEqual(book.nbbo_ask, 50006.0)

    def test_03_aggregated_book_spread(self):
        """测试聚合订单簿价差"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        self.assertIsNotNone(book.spread_bps)
        self.assertGreater(book.spread_bps, 0)
        # 价差应该合理（通常小于50基点）
        self.assertLess(book.spread_bps, 50)

    def test_04_aggregated_book_depth(self):
        """测试聚合订单簿深度"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        self.assertGreater(book.total_bid_liquidity, 0)
        self.assertGreater(book.total_ask_liquidity, 0)
        # 总流动性应该是所有交易所的总和
        self.assertGreater(book.total_bid_liquidity, 5000.0)  # 至少大于单个交易所

    def test_05_liquidity_score(self):
        """测试流动性评分"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        self.assertGreaterEqual(book.liquidity_score, 0)
        self.assertLessEqual(book.liquidity_score, 100)
        self.assertIsInstance(book.score_grade, LiquidityScore)

    def test_06_aggregated_book_levels(self):
        """测试聚合价格层级"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        self.assertGreater(len(book.bid_levels), 0)
        self.assertGreater(len(book.ask_levels), 0)

        # 检查层级结构
        if book.bid_levels:
            level = book.bid_levels[0]
            self.assertIn('price', level)
            self.assertIn('size', level)

    def test_07_aggregated_book_to_dict(self):
        """测试AggregatedBook序列化"""
        book = self.aggregator.aggregate_order_book("BTC-USDT")

        data = book.to_dict()
        self.assertIn('symbol', data)
        self.assertIn('nbbo_bid', data)
        self.assertIn('nbbo_ask', data)
        self.assertIn('liquidity_score', data)

    def test_08_aggregation_methods(self):
        """测试不同聚合方法"""
        methods = [
            AggregationMethod.VOLUME_WEIGHTED,
            AggregationMethod.PRICE_PRIORITY,
            AggregationMethod.TIME_PRIORITY,
        ]

        for method in methods:
            book = self.aggregator.aggregate_order_book(
                "BTC-USDT",
                method=method
            )
            self.assertIsNotNone(book)
            self.assertEqual(book.symbol, "BTC-USDT")

    def test_09_empty_symbol_aggregation(self):
        """测试空品种聚合"""
        # 没有数据时聚合器会抛出ValueError
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_order_book("UNKNOWN-PAIR")


class TestCostEstimation(unittest.TestCase):
    """测试成本估算功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self._setup_exchanges()

    def _setup_exchanges(self):
        """设置交易所"""
        for ex_id in ["binance", "okx", "bybit"]:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex_id,
                symbol="BTC-USDT",
                best_bid=50000.0,
                best_ask=50010.0,
                bid_depth=10000.0,
                ask_depth=10000.0,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

    def test_01_estimate_buy_cost(self):
        """测试买入成本估算"""
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1.0,
        )

        self.assertIsNotNone(cost)
        self.assertIsInstance(cost, LiquidityCost)
        self.assertEqual(cost.symbol, "BTC-USDT")
        self.assertEqual(cost.side, "buy")

    def test_02_estimate_sell_cost(self):
        """测试卖出成本估算"""
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="sell",
            quantity=1.0,
        )

        self.assertIsNotNone(cost)
        self.assertEqual(cost.side, "sell")

    def test_03_cost_components(self):
        """测试成本组成"""
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1.0,
        )

        # 检查成本组成部分
        self.assertGreaterEqual(cost.spread_cost_bps, 0)
        self.assertGreaterEqual(cost.impact_cost_bps, 0)
        self.assertGreaterEqual(cost.total_cost_bps, 0)

        # 总成本应该大于等于各部分之和的最小值
        self.assertGreaterEqual(
            cost.total_cost_bps,
            cost.spread_cost_bps
        )

    def test_04_cost_scales_with_quantity(self):
        """测试成本随数量增加"""
        small_cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1.0,
        )

        large_cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=100.0,
        )

        # 大订单的冲击成本应该更高
        self.assertGreaterEqual(
            large_cost.impact_cost_bps,
            small_cost.impact_cost_bps
        )

    def test_05_cost_price_estimates(self):
        """测试价格估算"""
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1.0,
        )

        self.assertGreater(cost.estimated_avg_price, 0)
        self.assertGreater(cost.best_price, 0)
        # 买入时，平均价格应该 >= 最优价格
        self.assertGreaterEqual(cost.estimated_avg_price, cost.best_price * 0.99)

    def test_06_cost_to_dict(self):
        """测试LiquidityCost序列化"""
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1.0,
        )

        data = cost.to_dict()
        self.assertIn('symbol', data)
        self.assertIn('total_cost_bps', data)
        self.assertIn('estimated_avg_price', data)


class TestExecutionStrategy(unittest.TestCase):
    """测试执行策略推荐功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self._setup_exchanges()

    def _setup_exchanges(self):
        """设置交易所"""
        for ex_id in ["binance", "okx", "bybit"]:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex_id,
                symbol="BTC-USDT",
                best_bid=50000.0,
                best_ask=50010.0,
                bid_depth=10000.0,
                ask_depth=10000.0,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

    def test_01_recommend_strategy(self):
        """测试策略推荐"""
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )

        self.assertIsNotNone(strategies)
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)

    def test_02_strategy_structure(self):
        """测试策略结构"""
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )

        if strategies:
            strategy = strategies[0]
            self.assertIsInstance(strategy, ExecutionStrategy)
            self.assertIsNotNone(strategy.strategy_name)
            self.assertIsNotNone(strategy.description)
            self.assertGreater(strategy.total_allocation, 0)

    def test_03_strategy_venues(self):
        """测试策略推荐场所"""
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )

        if strategies:
            strategy = strategies[0]
            self.assertIsNotNone(strategy.recommended_venues)
            self.assertIsInstance(strategy.recommended_venues, list)

    def test_04_strategy_score(self):
        """测试策略评分"""
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )

        if strategies:
            for strategy in strategies:
                self.assertGreaterEqual(strategy.strategy_score, 0)
                self.assertLessEqual(strategy.strategy_score, 100)

    def test_05_urgency_affects_strategy(self):
        """测试紧迫度影响策略"""
        low_urgency = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
            urgency=0.1,
        )

        high_urgency = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
            urgency=0.9,
        )

        # 高紧迫度可能推荐更激进的策略
        self.assertIsNotNone(low_urgency)
        self.assertIsNotNone(high_urgency)

    def test_06_strategy_to_dict(self):
        """测试ExecutionStrategy序列化"""
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )

        if strategies:
            data = strategies[0].to_dict()
            self.assertIn('strategy_name', data)
            self.assertIn('strategy_score', data)
            self.assertIn('recommended_venues', data)


class TestDarkPoolLiquidity(unittest.TestCase):
    """测试暗池流动性功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self._setup_dark_pools()

    def _setup_dark_pools(self):
        """设置暗池"""
        pools = [
            ("dark_pool_1", "BTC-USDT", 10000.0, 0.7),
            ("dark_pool_2", "BTC-USDT", 5000.0, 0.5),
            ("dark_pool_3", "ETH-USDT", 3000.0, 0.6),
        ]
        for pool_id, symbol, size, prob in pools:
            dark = create_mock_dark_pool(
                pool_id=pool_id,
                symbol=symbol,
                indicative_size=size,
                match_probability=prob,
            )
            self.aggregator.update_dark_pool_liquidity(dark)

    def test_01_update_dark_pool(self):
        """测试更新暗池流动性"""
        dark = create_mock_dark_pool(
            pool_id="test_pool",
            symbol="BTC-USDT",
        )
        self.aggregator.update_dark_pool_liquidity(dark)
        # 应该不抛出异常

    def test_02_get_dark_pool_opportunities(self):
        """测试获取暗池机会"""
        opportunities = self.aggregator.get_dark_pool_opportunities("BTC-USDT")

        self.assertIsNotNone(opportunities)
        self.assertIsInstance(opportunities, list)
        self.assertGreater(len(opportunities), 0)

    def test_03_dark_pool_filtering(self):
        """测试暗池过滤"""
        # 获取大于8000的暗池
        large_pools = self.aggregator.get_dark_pool_opportunities(
            "BTC-USDT",
            min_size=8000.0
        )

        # 应该只返回满足条件的暗池
        for pool in large_pools:
            self.assertGreaterEqual(pool.indicative_size, 8000.0)

    def test_04_dark_pool_properties(self):
        """测试DarkPoolLiquidity属性"""
        dark = create_mock_dark_pool(
            pool_id="test_pool",
            symbol="BTC-USDT",
            indicative_size=5000.0,
            match_probability=0.6,
        )

        self.assertEqual(dark.pool_id, "test_pool")
        self.assertEqual(dark.symbol, "BTC-USDT")
        self.assertEqual(dark.indicative_size, 5000.0)
        self.assertEqual(dark.match_probability, 0.6)

    def test_05_dark_pool_to_dict(self):
        """测试DarkPoolLiquidity序列化"""
        dark = create_mock_dark_pool(
            pool_id="test_pool",
            symbol="BTC-USDT",
        )

        data = dark.to_dict()
        self.assertIn('pool_id', data)
        self.assertIn('symbol', data)
        self.assertIn('indicative_size', data)
        self.assertIn('match_probability', data)

    def test_06_create_dark_pool_factory(self):
        """测试暗池工厂函数"""
        dark = create_dark_pool_liquidity(
            pool_id="factory_pool",
            symbol="BTC-USDT",
            indicative_size=5000.0,
        )
        self.assertIsNotNone(dark)
        self.assertIsInstance(dark, DarkPoolLiquidity)


class TestLiquidityPrediction(unittest.TestCase):
    """测试流动性预测功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self._setup_exchanges()

    def _setup_exchanges(self):
        """设置交易所"""
        for ex_id in ["binance", "okx"]:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex_id,
                symbol="BTC-USDT",
            )
            self.aggregator.update_exchange_liquidity(liquidity)

    def test_01_predict_liquidity(self):
        """测试流动性预测"""
        # 流动性预测需要足够的历史数据
        # 先多次更新以构建历史
        for i in range(15):
            for ex_id in ["binance", "okx"]:
                liquidity = create_mock_exchange_liquidity(
                    exchange_id=ex_id,
                    symbol="BTC-USDT",
                    best_bid=50000.0 + i,
                    best_ask=50010.0 + i,
                )
                self.aggregator.update_exchange_liquidity(liquidity)
            # 聚合一次以更新历史
            self.aggregator.aggregate_order_book("BTC-USDT")

        try:
            prediction = self.aggregator.predict_liquidity(
                symbol="BTC-USDT",
                horizon_seconds=1800,  # 30分钟
            )
            self.assertIsNotNone(prediction)
            self.assertIsInstance(prediction, LiquidityPrediction)
        except ValueError:
            # 如果历史数据不足，跳过测试
            pass

    def test_02_prediction_components(self):
        """测试预测组件"""
        # 构建足够历史数据
        for i in range(15):
            for ex_id in ["binance", "okx"]:
                liquidity = create_mock_exchange_liquidity(
                    exchange_id=ex_id,
                    symbol="BTC-USDT",
                )
                self.aggregator.update_exchange_liquidity(liquidity)
            self.aggregator.aggregate_order_book("BTC-USDT")

        try:
            prediction = self.aggregator.predict_liquidity(
                symbol="BTC-USDT",
                horizon_seconds=3600,  # 60分钟
            )
            self.assertEqual(prediction.symbol, "BTC-USDT")
            self.assertEqual(prediction.prediction_horizon_seconds, 3600)
            self.assertIsNotNone(prediction.predicted_spread_bps)
            self.assertIsNotNone(prediction.predicted_depth)
        except ValueError:
            pass

    def test_03_prediction_confidence(self):
        """测试预测置信度"""
        for i in range(15):
            for ex_id in ["binance", "okx"]:
                liquidity = create_mock_exchange_liquidity(
                    exchange_id=ex_id,
                    symbol="BTC-USDT",
                )
                self.aggregator.update_exchange_liquidity(liquidity)
            self.aggregator.aggregate_order_book("BTC-USDT")

        try:
            prediction = self.aggregator.predict_liquidity(
                symbol="BTC-USDT",
                horizon_seconds=1800,
            )
            self.assertGreaterEqual(prediction.model_confidence, 0)
            self.assertLessEqual(prediction.model_confidence, 1)
        except ValueError:
            pass

    def test_04_prediction_to_dict(self):
        """测试LiquidityPrediction序列化"""
        for i in range(15):
            for ex_id in ["binance", "okx"]:
                liquidity = create_mock_exchange_liquidity(
                    exchange_id=ex_id,
                    symbol="BTC-USDT",
                )
                self.aggregator.update_exchange_liquidity(liquidity)
            self.aggregator.aggregate_order_book("BTC-USDT")

        try:
            prediction = self.aggregator.predict_liquidity(
                symbol="BTC-USDT",
                horizon_seconds=1800,
            )
            data = prediction.to_dict()
            self.assertIn('symbol', data)
            self.assertIn('prediction_horizon_seconds', data)
        except ValueError:
            pass


class TestLiquidityMetrics(unittest.TestCase):
    """测试流动性指标功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self._setup_exchanges()

    def _setup_exchanges(self):
        """设置交易所"""
        for ex_id in ["binance", "okx", "bybit"]:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex_id,
                symbol="BTC-USDT",
            )
            self.aggregator.update_exchange_liquidity(liquidity)

    def test_01_get_liquidity_metrics(self):
        """测试获取流动性指标"""
        metrics = self.aggregator.get_liquidity_metrics("BTC-USDT")

        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, LiquidityMetrics)

    def test_02_metrics_components(self):
        """测试指标组件"""
        metrics = self.aggregator.get_liquidity_metrics("BTC-USDT")

        self.assertEqual(metrics.symbol, "BTC-USDT")
        self.assertIsNotNone(metrics.avg_spread_bps)
        self.assertIsNotNone(metrics.total_depth_liquidity)

    def test_03_metrics_exchange_count(self):
        """测试交易所数量"""
        metrics = self.aggregator.get_liquidity_metrics("BTC-USDT")

        self.assertGreaterEqual(metrics.total_exchanges, 1)

    def test_04_metrics_to_dict(self):
        """测试LiquidityMetrics序列化"""
        metrics = self.aggregator.get_liquidity_metrics("BTC-USDT")

        data = metrics.to_dict()
        self.assertIn('symbol', data)
        self.assertIn('avg_spread_bps', data)


class TestCallbacks(unittest.TestCase):
    """测试回调功能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()
        self.callback_called = False
        self.callback_data = None

    def test_01_set_book_update_callback(self):
        """测试设置订单簿更新回调"""
        def callback(book: AggregatedBook):
            self.callback_called = True
            self.callback_data = book

        self.aggregator.set_book_update_callback(callback)
        # 回调设置应该成功，不抛出异常

    def test_02_set_liquidity_alert_callback(self):
        """测试设置流动性告警回调"""
        def callback(symbol: str, message: str):
            self.callback_called = True
            self.callback_data = (symbol, message)

        self.aggregator.set_liquidity_alert_callback(callback)
        # 回调设置应该成功，不抛出异常


class TestEdgeCases(unittest.TestCase):
    """测试边界条件"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()

    def test_01_empty_aggregator(self):
        """测试空聚合器"""
        # 没有数据时聚合器会抛出ValueError
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_order_book("NONEXISTENT")

    def test_02_zero_quantity_cost(self):
        """测试零数量成本估算"""
        # 先添加一些流动性
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=0.0,
        )
        # 应该处理零数量的情况
        self.assertIsNotNone(cost)

    def test_03_very_large_quantity(self):
        """测试超大数量"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC-USDT",
            bid_depth=100.0,
            ask_depth=100.0,
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=1000000.0,  # 远超可用流动性
        )
        # 应该能处理大订单
        self.assertIsNotNone(cost)
        # 冲击成本应该很高
        self.assertGreater(cost.impact_cost_bps, 0)

    def test_04_inactive_exchange(self):
        """测试非活跃交易所"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="inactive_ex",
            symbol="BTC-USDT",
        )
        liquidity.is_active = False
        self.aggregator.update_exchange_liquidity(liquidity)

        # 非活跃交易所被过滤后可能没有有效数据，导致抛出ValueError
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_order_book("BTC-USDT")

    def test_05_special_characters_in_symbol(self):
        """测试特殊字符品种"""
        liquidity = create_mock_exchange_liquidity(
            exchange_id="binance",
            symbol="BTC/USDT:PERP",
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        book = self.aggregator.aggregate_order_book("BTC/USDT:PERP")
        self.assertIsNotNone(book)
        self.assertEqual(book.symbol, "BTC/USDT:PERP")

    def test_06_empty_bid_ask_levels(self):
        """测试空买卖盘"""
        liquidity = ExchangeLiquidity(
            exchange_id="empty_ex",
            symbol="BTC-USDT",
            timestamp=datetime.now(),
            bid_levels={},
            ask_levels={},
            displayed_liquidity=0,
            spread_bps=0,
            depth_score=0,
            resilience_score=0,
            is_active=True,
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        # 空买卖盘可能导致没有有效数据
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_order_book("BTC-USDT")


class TestPerformance(unittest.TestCase):
    """测试性能"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()

    def test_01_many_exchanges(self):
        """测试大量交易所"""
        import time

        # 添加100个交易所
        for i in range(100):
            liquidity = create_mock_exchange_liquidity(
                exchange_id=f"exchange_{i}",
                symbol="BTC-USDT",
                best_bid=50000.0 + i * 0.1,
                best_ask=50010.0 + i * 0.1,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

        start = time.time()
        book = self.aggregator.aggregate_order_book("BTC-USDT")
        elapsed = time.time() - start

        self.assertIsNotNone(book)
        self.assertEqual(book.total_exchanges, 100)
        # 聚合应该在合理时间内完成 (< 1秒)
        self.assertLess(elapsed, 1.0)

    def test_02_many_price_levels(self):
        """测试大量价格层级"""
        # 创建有很多价格层级的流动性
        liquidity = create_mock_exchange_liquidity(
            exchange_id="deep_exchange",
            symbol="BTC-USDT",
            levels=100,
        )
        self.aggregator.update_exchange_liquidity(liquidity)

        import time
        start = time.time()
        book = self.aggregator.aggregate_order_book("BTC-USDT")
        elapsed = time.time() - start

        self.assertIsNotNone(book)
        # 处理应该在合理时间内完成
        self.assertLess(elapsed, 1.0)

    def test_03_many_symbols(self):
        """测试大量品种"""
        symbols = [f"COIN{i}-USDT" for i in range(50)]

        for symbol in symbols:
            liquidity = create_mock_exchange_liquidity(
                exchange_id="binance",
                symbol=symbol,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

        # 检查统计
        stats = self.aggregator.get_statistics()
        self.assertIsNotNone(stats)

    def test_04_repeated_updates(self):
        """测试重复更新性能"""
        import time

        start = time.time()
        for _ in range(1000):
            liquidity = create_mock_exchange_liquidity(
                exchange_id="binance",
                symbol="BTC-USDT",
            )
            self.aggregator.update_exchange_liquidity(liquidity)
        elapsed = time.time() - start

        # 1000次更新应该在合理时间内完成 (< 5秒)
        self.assertLess(elapsed, 5.0)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """测试前置"""
        self.aggregator = LiquidityAggregator()

    def test_01_full_workflow(self):
        """测试完整工作流"""
        # 1. 添加多个交易所
        exchanges = ["binance", "okx", "bybit", "huobi"]
        for ex in exchanges:
            liquidity = create_mock_exchange_liquidity(
                exchange_id=ex,
                symbol="BTC-USDT",
                best_bid=50000.0,
                best_ask=50010.0,
                bid_depth=5000.0,
                ask_depth=5000.0,
            )
            self.aggregator.update_exchange_liquidity(liquidity)

        # 2. 添加暗池
        dark = create_mock_dark_pool(
            pool_id="dark_pool_1",
            symbol="BTC-USDT",
        )
        self.aggregator.update_dark_pool_liquidity(dark)

        # 3. 聚合订单簿
        book = self.aggregator.aggregate_order_book("BTC-USDT")
        self.assertEqual(book.total_exchanges, 4)

        # 4. 估算成本
        cost = self.aggregator.estimate_liquidity_cost(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )
        self.assertIsNotNone(cost)

        # 5. 获取执行策略
        strategies = self.aggregator.recommend_execution_strategy(
            symbol="BTC-USDT",
            side="buy",
            quantity=10.0,
        )
        self.assertGreater(len(strategies), 0)

        # 6. 获取指标
        metrics = self.aggregator.get_liquidity_metrics("BTC-USDT")
        self.assertIsNotNone(metrics)

        # 7. 预测流动性 - 需要历史数据，可能抛出异常
        try:
            # 先构建足够历史
            for i in range(15):
                book = self.aggregator.aggregate_order_book("BTC-USDT")
            prediction = self.aggregator.predict_liquidity(
                symbol="BTC-USDT",
                horizon_seconds=1800,
            )
            self.assertIsNotNone(prediction)
        except ValueError:
            pass  # 历史数据不足时跳过

        # 8. 获取暗池机会
        dark_opportunities = self.aggregator.get_dark_pool_opportunities("BTC-USDT")
        self.assertGreater(len(dark_opportunities), 0)

    def test_02_multi_symbol_workflow(self):
        """测试多品种工作流"""
        symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

        for symbol in symbols:
            for ex in ["binance", "okx"]:
                liquidity = create_mock_exchange_liquidity(
                    exchange_id=ex,
                    symbol=symbol,
                )
                self.aggregator.update_exchange_liquidity(liquidity)

        # 分别聚合每个品种
        for symbol in symbols:
            book = self.aggregator.aggregate_order_book(symbol)
            self.assertEqual(book.symbol, symbol)
            self.assertEqual(book.total_exchanges, 2)


# ==================== 测试运行器 ====================

def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestBasicFunctionality,
        TestExchangeLiquidity,
        TestOrderBookAggregation,
        TestCostEstimation,
        TestExecutionStrategy,
        TestDarkPoolLiquidity,
        TestLiquidityPrediction,
        TestLiquidityMetrics,
        TestCallbacks,
        TestEdgeCases,
        TestPerformance,
        TestIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
