"""
智能订单路由测试

测试:
- 场所管理
- 报价聚合
- 路由策略
- 订单执行
- 成交处理
- 执行质量分析
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.smart_order_router import (
    SmartOrderRouter,
    Venue,
    VenueQuote,
    RouteOrder,
    ChildOrder,
    RouteDecision,
    ExecutionQuality,
    SORConfig,
    VenueType,
    OrderType,
    OrderSide,
    RouteStrategy,
    RouteStatus,
    BestPriceStrategy,
    BestExecutionStrategy,
    LiquiditySeekingStrategy,
    DarkFirstStrategy,
    SprayStrategy,
    create_smart_router,
    create_venue,
    create_route_order,
    create_venue_quote,
    quick_route,
)


# ==================== Fixtures ====================

@pytest.fixture
def sor_config():
    """SOR配置"""
    return SORConfig(
        default_strategy=RouteStrategy.BEST_EXECUTION,
        max_venues_per_order=5,
        dark_pool_enabled=True,
        max_dark_pool_pct=0.3,
    )


@pytest.fixture
def sample_venues():
    """样本执行场所"""
    return [
        Venue(
            venue_id="NYSE",
            name="New York Stock Exchange",
            venue_type=VenueType.EXCHANGE,
            taker_fee_bps=3,
            maker_fee_bps=2,
            fill_rate=0.9,
            priority=10,
        ),
        Venue(
            venue_id="NASDAQ",
            name="NASDAQ",
            venue_type=VenueType.EXCHANGE,
            taker_fee_bps=2.5,
            maker_fee_bps=1.5,
            fill_rate=0.85,
            priority=9,
        ),
        Venue(
            venue_id="BATS",
            name="BATS Exchange",
            venue_type=VenueType.ECN,
            taker_fee_bps=2,
            rebate_bps=1,
            fill_rate=0.8,
            priority=7,
        ),
        Venue(
            venue_id="SIGMA",
            name="Sigma X Dark Pool",
            venue_type=VenueType.DARK_POOL,
            taker_fee_bps=1,
            fill_rate=0.5,
            supports_dark=True,
            supports_midpoint=True,
            priority=5,
        ),
    ]


@pytest.fixture
def sample_quotes():
    """样本报价"""
    return {
        "NYSE": VenueQuote(
            venue_id="NYSE",
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=5000,
            ask_price=150.05,
            ask_size=4000,
        ),
        "NASDAQ": VenueQuote(
            venue_id="NASDAQ",
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.01,
            bid_size=3000,
            ask_price=150.04,
            ask_size=3500,
        ),
        "BATS": VenueQuote(
            venue_id="BATS",
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.99,
            bid_size=2000,
            ask_price=150.06,
            ask_size=2500,
        ),
        "SIGMA": VenueQuote(
            venue_id="SIGMA",
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.02,
            bid_size=10000,
            ask_price=150.03,
            ask_size=10000,
            hidden_liquidity=5000,
        ),
    }


@pytest.fixture
def router(sor_config, sample_venues, sample_quotes):
    """配置好的路由器"""
    r = SmartOrderRouter(sor_config)
    for venue in sample_venues:
        r.register_venue(venue)
    for quote in sample_quotes.values():
        r.update_quote(quote)
    return r


# ==================== 场所管理测试 ====================

class TestVenueManagement:
    """场所管理测试"""

    def test_create_venue(self):
        """测试创建场所"""
        venue = create_venue(
            venue_id="TEST",
            name="Test Exchange",
            venue_type="exchange",
            taker_fee_bps=2.5,
        )

        assert venue.venue_id == "TEST"
        assert venue.venue_type == VenueType.EXCHANGE
        assert venue.taker_fee_bps == 2.5

    def test_register_venue(self):
        """测试注册场所"""
        router = SmartOrderRouter()
        venue = Venue(
            venue_id="TEST",
            name="Test",
            venue_type=VenueType.EXCHANGE,
        )

        router.register_venue(venue)

        assert "TEST" in router._venues
        assert router._venues["TEST"].name == "Test"

    def test_unregister_venue(self):
        """测试注销场所"""
        router = SmartOrderRouter()
        venue = Venue(venue_id="TEST", name="Test", venue_type=VenueType.EXCHANGE)
        router.register_venue(venue)
        router.unregister_venue("TEST")

        assert "TEST" not in router._venues

    def test_get_venues(self, router):
        """测试获取场所列表"""
        venues = router.get_venues()

        assert len(venues) == 4
        venue_ids = [v.venue_id for v in venues]
        assert "NYSE" in venue_ids
        assert "SIGMA" in venue_ids


# ==================== 报价管理测试 ====================

class TestQuoteManagement:
    """报价管理测试"""

    def test_create_venue_quote(self):
        """测试创建报价"""
        quote = create_venue_quote(
            venue_id="NYSE",
            symbol="AAPL",
            bid_price=150.0,
            bid_size=1000,
            ask_price=150.1,
            ask_size=800,
        )

        assert quote.venue_id == "NYSE"
        assert quote.mid_price == 150.05
        assert quote.spread_bps > 0

    def test_update_quote(self):
        """测试更新报价"""
        router = SmartOrderRouter()
        quote = VenueQuote(
            venue_id="NYSE",
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            bid_size=1000,
            ask_price=150.1,
            ask_size=800,
        )

        router.update_quote(quote)

        assert "AAPL" in router._quotes
        assert "NYSE" in router._quotes["AAPL"]

    def test_consolidated_book(self, router):
        """测试合并订单簿"""
        book = router.get_consolidated_book("AAPL")

        assert book['symbol'] == "AAPL"
        assert len(book['bids']) > 0
        assert len(book['asks']) > 0
        assert 'nbbo' in book
        assert book['venues'] == 4

    def test_nbbo_calculation(self, router):
        """测试NBBO计算"""
        book = router.get_consolidated_book("AAPL")
        nbbo = book['nbbo']

        # SIGMA有最高买价150.02
        assert nbbo['best_bid'] == 150.02
        # SIGMA有最低卖价150.03
        assert nbbo['best_ask'] == 150.03


# ==================== 路由策略测试 ====================

class TestBestPriceStrategy:
    """最优价格策略测试"""

    def test_calculate_route_buy(self, sample_venues, sample_quotes):
        """测试买入路由"""
        strategy = BestPriceStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.LIMIT,
            strategy=RouteStrategy.BEST_PRICE,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        assert len(decisions) > 0
        # 应该优先选择最低卖价
        assert decisions[0].venue_id == "SIGMA"  # 150.03最低

    def test_calculate_route_sell(self, sample_venues, sample_quotes):
        """测试卖出路由"""
        strategy = BestPriceStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=3000,
            order_type=OrderType.LIMIT,
            strategy=RouteStrategy.BEST_PRICE,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        assert len(decisions) > 0
        # 应该优先选择最高买价
        assert decisions[0].venue_id == "SIGMA"  # 150.02最高


class TestBestExecutionStrategy:
    """最佳执行策略测试"""

    def test_calculate_route(self, sample_venues, sample_quotes):
        """测试综合评分路由"""
        strategy = BestExecutionStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.LIMIT,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        assert len(decisions) > 0
        # 每个决策都有评分原因
        assert all('Score' in d.reasoning for d in decisions)

    def test_considers_multiple_factors(self, sample_venues, sample_quotes):
        """测试考虑多个因素"""
        strategy = BestExecutionStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=20000,  # 更大的订单量需要多个场所
            order_type=OrderType.LIMIT,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        # 应该分配到至少一个场所
        assert len(decisions) >= 1


class TestLiquiditySeekingStrategy:
    """流动性寻找策略测试"""

    def test_prioritize_liquidity(self, sample_venues, sample_quotes):
        """测试优先流动性"""
        strategy = LiquiditySeekingStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=8000,
            order_type=OrderType.LIMIT,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        assert len(decisions) > 0
        # SIGMA有最大流动性(10000 + 5000隐含)
        assert decisions[0].venue_id == "SIGMA"


class TestDarkFirstStrategy:
    """暗池优先策略测试"""

    def test_dark_pool_priority(self, sample_venues, sample_quotes):
        """测试暗池优先"""
        strategy = DarkFirstStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=5000,
            order_type=OrderType.LIMIT,
            max_dark_pct=0.5,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        assert len(decisions) > 0
        # 首个决策应该是暗池
        assert decisions[0].venue_id == "SIGMA"
        assert "Dark pool" in decisions[0].reasoning

    def test_respects_dark_limit(self, sample_venues, sample_quotes):
        """测试暗池限制"""
        strategy = DarkFirstStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000,
            order_type=OrderType.LIMIT,
            max_dark_pct=0.2,  # 最多20%到暗池
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        # 暗池分配不应超过2000
        dark_allocation = sum(
            d.allocation for d in decisions
            if d.venue_id == "SIGMA"
        )
        assert dark_allocation <= 2000


class TestSprayStrategy:
    """喷射策略测试"""

    def test_distribute_to_venues(self, sample_venues, sample_quotes):
        """测试分散到多个场所"""
        strategy = SprayStrategy()
        order = RouteOrder(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=8000,
            order_type=OrderType.LIMIT,
            max_venues=4,
        )

        decisions = strategy.calculate_route(order, sample_venues, sample_quotes)

        # 应该分配到多个场所
        assert len(decisions) >= 3
        # 分配应该相对均匀
        venues_used = set(d.venue_id for d in decisions)
        assert len(venues_used) >= 3


# ==================== 订单路由测试 ====================

class TestOrderRouting:
    """订单路由测试"""

    @pytest.mark.asyncio
    async def test_route_order(self, router):
        """测试路由订单"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=5000,
            strategy="best_execution",
            limit_price=150.10,
        )

        result = await router.route_order(order)

        assert result.status == RouteStatus.ROUTED
        assert len(result.child_orders) > 0
        assert result.routed_at is not None

    @pytest.mark.asyncio
    async def test_route_with_venue_filter(self, router):
        """测试带场所过滤的路由"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=3000,
        )
        order.allowed_venues = ["NYSE", "NASDAQ"]

        result = await router.route_order(order)

        # 只应该路由到允许的场所
        for venue_id in result.venue_allocations:
            assert venue_id in ["NYSE", "NASDAQ"]

    @pytest.mark.asyncio
    async def test_route_with_excluded_venues(self, router):
        """测试排除场所"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=3000,
        )
        order.excluded_venues = ["SIGMA"]

        result = await router.route_order(order)

        assert "SIGMA" not in result.venue_allocations

    @pytest.mark.asyncio
    async def test_route_validation(self, router):
        """测试订单验证"""
        # 数量为0
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=0,
        )

        with pytest.raises(ValueError):
            await router.route_order(order)

    @pytest.mark.asyncio
    async def test_route_no_quotes(self):
        """测试无报价时路由"""
        router = SmartOrderRouter()
        venue = Venue(venue_id="TEST", name="Test", venue_type=VenueType.EXCHANGE)
        router.register_venue(venue)

        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
        )

        with pytest.raises(ValueError, match="No quotes"):
            await router.route_order(order)


# ==================== 成交处理测试 ====================

class TestFillProcessing:
    """成交处理测试"""

    @pytest.mark.asyncio
    async def test_process_fill(self, router):
        """测试处理成交"""
        # 先路由订单
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=5000,
        )
        routed = await router.route_order(order)
        child_id = routed.child_orders[0]

        # 处理成交
        child, parent = await router.process_fill(
            child_id=child_id,
            filled_quantity=1000,
            filled_price=150.05,
            fees=0.5,
        )

        assert child.filled_quantity == 1000
        assert child.filled_price == 150.05
        assert parent.filled_quantity == 1000
        assert parent.total_fees == 0.5

    @pytest.mark.asyncio
    async def test_partial_fill(self, router):
        """测试部分成交"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=5000,
        )
        routed = await router.route_order(order)
        child_id = routed.child_orders[0]
        child = router.get_child_order(child_id)
        original_qty = child.quantity

        # 部分成交
        await router.process_fill(child_id, original_qty / 2, 150.05, 0.25)

        assert routed.status == RouteStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_full_fill(self, router):
        """测试完全成交"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            strategy="spray",
        )
        order.max_venues = 1  # 只用一个场所
        routed = await router.route_order(order)

        # 完全成交
        for child_id in routed.child_orders:
            child = router.get_child_order(child_id)
            await router.process_fill(child_id, child.quantity, 150.05, 0.5)

        updated = router.get_order(routed.order_id)
        assert updated.status == RouteStatus.FILLED

    @pytest.mark.asyncio
    async def test_fill_not_found(self, router):
        """测试成交不存在的子订单"""
        with pytest.raises(ValueError, match="not found"):
            await router.process_fill("NONEXISTENT", 100, 150.0, 0)


# ==================== 取消订单测试 ====================

class TestOrderCancellation:
    """取消订单测试"""

    @pytest.mark.asyncio
    async def test_cancel_order(self, router):
        """测试取消订单"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=5000,
        )
        routed = await router.route_order(order)

        cancelled = await router.cancel_order(routed.order_id)

        assert cancelled.status == RouteStatus.CANCELLED
        assert cancelled.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_completed_order(self, router):
        """测试取消已完成订单"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            strategy="spray",
        )
        order.max_venues = 1
        routed = await router.route_order(order)

        # 完成订单
        for child_id in routed.child_orders:
            child = router.get_child_order(child_id)
            await router.process_fill(child_id, child.quantity, 150.05, 0)

        with pytest.raises(ValueError, match="already complete"):
            await router.cancel_order(routed.order_id)


# ==================== 执行质量分析测试 ====================

class TestExecutionQuality:
    """执行质量分析测试"""

    @pytest.mark.asyncio
    async def test_analyze_execution(self, router):
        """测试执行质量分析"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=2000,
            limit_price=150.0,
        )
        routed = await router.route_order(order)

        # 模拟成交
        for child_id in routed.child_orders:
            child = router.get_child_order(child_id)
            await router.process_fill(child_id, child.quantity, 150.05, 1.0)

        # 分析执行质量
        quality = router.analyze_execution(routed.order_id)

        assert quality.order_id == routed.order_id
        assert quality.filled_quantity == 2000
        assert quality.implementation_shortfall_bps != 0
        assert 'venue_breakdown' in quality.to_dict()
        assert quality.execution_score >= 0

    @pytest.mark.asyncio
    async def test_execution_score(self, router):
        """测试执行评分"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
        )
        routed = await router.route_order(order)

        for child_id in routed.child_orders:
            child = router.get_child_order(child_id)
            await router.process_fill(child_id, child.quantity, 150.03, 0.3)

        quality = router.analyze_execution(routed.order_id)

        # 评分应该在合理范围
        assert 0 <= quality.execution_score <= 100


# ==================== 统计测试 ====================

class TestStatistics:
    """统计测试"""

    @pytest.mark.asyncio
    async def test_get_statistics(self, router):
        """测试获取统计"""
        # 路由一些订单
        for _ in range(3):
            order = create_route_order(
                symbol="AAPL",
                side="buy",
                quantity=1000,
            )
            await router.route_order(order)

        stats = router.get_statistics()

        assert stats['orders_routed'] == 3
        assert stats['total_volume'] == 3000
        assert stats['active_venues'] == 4


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_smart_router(self):
        """测试创建路由器"""
        router = create_smart_router(
            default_strategy="best_price",
            dark_pool_enabled=False,
        )

        assert router.config.default_strategy == RouteStrategy.BEST_PRICE
        assert router.config.dark_pool_enabled is False

    def test_create_venue(self):
        """测试创建场所"""
        venue = create_venue(
            venue_id="TEST",
            name="Test Exchange",
            venue_type="exchange",
            taker_fee_bps=2.5,
            fill_rate=0.9,
        )

        assert venue.venue_id == "TEST"
        assert venue.taker_fee_bps == 2.5
        assert venue.fill_rate == 0.9

    def test_create_route_order(self):
        """测试创建路由订单"""
        order = create_route_order(
            symbol="AAPL",
            side="sell",
            quantity=5000,
            strategy="liquidity",
            limit_price=150.0,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.strategy == RouteStrategy.LIQUIDITY_SEEKING

    @pytest.mark.asyncio
    async def test_quick_route(self):
        """测试快速路由"""
        venues = [
            {"venue_id": "V1", "venue_type": "exchange", "fill_rate": 0.9},
            {"venue_id": "V2", "venue_type": "exchange", "fill_rate": 0.85},
        ]
        quotes = [
            {"venue_id": "V1", "bid_price": 100, "bid_size": 1000, "ask_price": 100.1, "ask_size": 800},
            {"venue_id": "V2", "bid_price": 99.9, "bid_size": 1500, "ask_price": 100.05, "ask_size": 1200},
        ]

        order = await quick_route("TEST", "buy", 500, venues, quotes)

        assert order.status == RouteStatus.ROUTED
        assert len(order.child_orders) > 0


# ==================== 回调测试 ====================

class TestCallbacks:
    """回调测试"""

    @pytest.mark.asyncio
    async def test_child_created_callback(self, router):
        """测试子订单创建回调"""
        callback = MagicMock()
        router.set_child_callback(callback)

        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
        )
        await router.route_order(order)

        assert callback.called

    @pytest.mark.asyncio
    async def test_completion_callback(self, router):
        """测试完成回调"""
        callback = MagicMock()
        router.set_completion_callback(callback)

        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            strategy="spray",
        )
        order.max_venues = 1
        routed = await router.route_order(order)

        # 完成订单
        for child_id in routed.child_orders:
            child = router.get_child_order(child_id)
            await router.process_fill(child_id, child.quantity, 150.0, 0)

        assert callback.called


# ==================== 边界条件测试 ====================

class TestEdgeCases:
    """边界条件测试"""

    def test_venue_to_dict(self, sample_venues):
        """测试场所转字典"""
        venue = sample_venues[0]
        d = venue.to_dict()

        assert d['venue_id'] == venue.venue_id
        assert d['venue_type'] == venue.venue_type.value

    def test_quote_to_dict(self, sample_quotes):
        """测试报价转字典"""
        quote = sample_quotes['NYSE']
        d = quote.to_dict()

        assert d['venue_id'] == 'NYSE'
        assert 'mid_price' in d
        assert 'spread_bps' in d

    def test_order_to_dict(self):
        """测试订单转字典"""
        order = create_route_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
        )
        d = order.to_dict()

        assert d['symbol'] == 'AAPL'
        assert d['side'] == 'buy'
        assert 'fill_rate' in d

    def test_route_decision_to_dict(self):
        """测试路由决策转字典"""
        decision = RouteDecision(
            venue_id="NYSE",
            allocation=1000,
            expected_fill_rate=0.9,
            expected_cost_bps=2.5,
            reasoning="Test",
        )
        d = decision.to_dict()

        assert d['venue_id'] == 'NYSE'
        assert d['expected_fill_rate'] == 90.0

    def test_stale_quote_filtering(self, router, sample_venues):
        """测试过期报价过滤"""
        # 标记报价为过期
        stale_quote = VenueQuote(
            venue_id="NYSE",
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(seconds=60),
            bid_price=150.0,
            bid_size=1000,
            ask_price=150.1,
            ask_size=800,
            is_stale=True,
        )
        router.update_quote(stale_quote)

        book = router.get_consolidated_book("AAPL")

        # 过期报价应被过滤
        nyse_bids = [b for b in book['bids'] if b['venue'] == 'NYSE']
        assert len(nyse_bids) == 0


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
