"""
交易所适配器基类测试

测试内容:
- 枚举类型
- 数据类
- 配置类
- 异常类
"""

import pytest
from datetime import datetime

from quantrisk_realtime.exchanges import (
    # Enums
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    # Data classes
    Order,
    Position,
    Balance,
    AccountInfo,
    Trade,
    Ticker,
    # Config
    ExchangeConfig,
    # Exceptions
    ExchangeError,
    OrderError,
    InsufficientFundsError,
)
from quantrisk_realtime.exchanges.base import PositionSide, RateLimitError


# ==================== Enum Tests ====================

class TestOrderStatus:
    """订单状态枚举测试"""

    def test_all_values(self):
        """测试所有状态值"""
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.NEW == "new"
        assert OrderStatus.PARTIALLY_FILLED == "partially_filled"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.CANCELED == "canceled"
        assert OrderStatus.REJECTED == "rejected"
        assert OrderStatus.EXPIRED == "expired"

    def test_string_conversion(self):
        """测试字符串转换"""
        assert OrderStatus.FILLED.value == "filled"


class TestOrderType:
    """订单类型枚举测试"""

    def test_basic_types(self):
        """测试基本类型"""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"

    def test_stop_types(self):
        """测试止损类型"""
        assert OrderType.STOP_MARKET == "stop_market"
        assert OrderType.STOP_LIMIT == "stop_limit"
        assert OrderType.TRAILING_STOP == "trailing_stop"


class TestOrderSide:
    """订单方向枚举测试"""

    def test_sides(self):
        """测试买卖方向"""
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"


class TestTimeInForce:
    """订单有效期枚举测试"""

    def test_all_values(self):
        """测试所有值"""
        assert TimeInForce.GTC == "gtc"
        assert TimeInForce.IOC == "ioc"
        assert TimeInForce.FOK == "fok"
        assert TimeInForce.GTX == "gtx"


class TestPositionSide:
    """持仓方向枚举测试"""

    def test_all_values(self):
        """测试所有值"""
        assert PositionSide.LONG == "long"
        assert PositionSide.SHORT == "short"
        assert PositionSide.BOTH == "both"


# ==================== Data Class Tests ====================

class TestOrder:
    """订单数据类测试"""

    def test_creation(self):
        """测试创建订单"""
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
        )
        assert order.order_id == "ORD-001"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == 1.0
        assert order.price == 50000.0

    def test_default_values(self):
        """测试默认值"""
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.avg_fill_price == 0.0
        assert order.time_in_force == TimeInForce.GTC

    def test_is_open(self):
        """测试is_open属性"""
        # 新订单
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.NEW,
        )
        assert order.is_open

        # 已成交订单
        order.status = OrderStatus.FILLED
        assert not order.is_open

        # 已取消订单
        order.status = OrderStatus.CANCELED
        assert not order.is_open

    def test_is_filled(self):
        """测试is_filled属性"""
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
        )
        assert order.is_filled

    def test_remaining_quantity(self):
        """测试剩余数量"""
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            filled_quantity=3.0,
        )
        assert order.remaining_quantity == 7.0

    def test_to_dict(self):
        """测试转字典"""
        order = Order(
            order_id="ORD-001",
            client_order_id="CLI-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.NEW,
        )
        d = order.to_dict()
        assert d['order_id'] == "ORD-001"
        assert d['symbol'] == "BTC/USDT"
        assert d['side'] == "buy"
        assert d['order_type'] == "limit"
        assert d['is_open'] == True


class TestPosition:
    """持仓数据类测试"""

    def test_creation(self):
        """测试创建持仓"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.5,
            entry_price=50000.0,
            mark_price=52000.0,
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.side == PositionSide.LONG
        assert pos.quantity == 0.5
        assert pos.entry_price == 50000.0

    def test_notional_value(self):
        """测试名义价值"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.5,
            entry_price=50000.0,
            mark_price=52000.0,
        )
        assert pos.notional_value == 0.5 * 52000.0

    def test_pnl_pct_long(self):
        """测试多头盈亏百分比"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            mark_price=52000.0,
        )
        # (52000 - 50000) / 50000 * 100 = 4%
        assert pos.pnl_pct == pytest.approx(4.0)

    def test_pnl_pct_short(self):
        """测试空头盈亏百分比"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            quantity=1.0,
            entry_price=50000.0,
            mark_price=48000.0,
        )
        # (50000 - 48000) / 50000 * 100 = 4%
        assert pos.pnl_pct == pytest.approx(4.0)

    def test_pnl_pct_zero_entry(self):
        """测试入场价为0的情况"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=0.0,
            mark_price=50000.0,
        )
        assert pos.pnl_pct == 0

    def test_to_dict(self):
        """测试转字典"""
        pos = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.5,
            entry_price=50000.0,
            mark_price=52000.0,
            leverage=10,
        )
        d = pos.to_dict()
        assert d['symbol'] == "BTC/USDT"
        assert d['side'] == "long"
        assert d['leverage'] == 10
        assert 'notional_value' in d
        assert 'pnl_pct' in d


class TestBalance:
    """余额数据类测试"""

    def test_creation(self):
        """测试创建余额"""
        balance = Balance(
            asset="USDT",
            free=10000.0,
            locked=500.0,
        )
        assert balance.asset == "USDT"
        assert balance.free == 10000.0
        assert balance.locked == 500.0
        assert balance.total == 10500.0  # auto calculated

    def test_explicit_total(self):
        """测试显式指定total"""
        balance = Balance(
            asset="BTC",
            free=1.0,
            locked=0.5,
            total=1.5,
        )
        assert balance.total == 1.5

    def test_to_dict(self):
        """测试转字典"""
        balance = Balance(
            asset="USDT",
            free=10000.0,
            locked=500.0,
        )
        d = balance.to_dict()
        assert d['asset'] == "USDT"
        assert d['free'] == 10000.0
        assert d['total'] == 10500.0


class TestAccountInfo:
    """账户信息数据类测试"""

    def test_creation(self):
        """测试创建账户信息"""
        balances = [
            Balance(asset="USDT", free=10000.0, locked=0.0),
            Balance(asset="BTC", free=0.5, locked=0.0),
        ]
        account = AccountInfo(
            account_id="ACC-001",
            balances=balances,
            total_wallet_balance=15000.0,
        )
        assert account.account_id == "ACC-001"
        assert len(account.balances) == 2

    def test_get_balance(self):
        """测试获取余额"""
        balances = [
            Balance(asset="USDT", free=10000.0, locked=0.0),
            Balance(asset="BTC", free=0.5, locked=0.0),
        ]
        account = AccountInfo(
            account_id="ACC-001",
            balances=balances,
        )

        usdt = account.get_balance("USDT")
        assert usdt is not None
        assert usdt.free == 10000.0

        eth = account.get_balance("ETH")
        assert eth is None

    def test_to_dict(self):
        """测试转字典"""
        balances = [
            Balance(asset="USDT", free=10000.0, locked=0.0),
        ]
        account = AccountInfo(
            account_id="ACC-001",
            balances=balances,
            can_trade=True,
        )
        d = account.to_dict()
        assert d['account_id'] == "ACC-001"
        assert d['can_trade'] == True
        assert len(d['balances']) == 1


class TestTrade:
    """成交记录数据类测试"""

    def test_creation(self):
        """测试创建成交记录"""
        trade = Trade(
            trade_id="TRD-001",
            order_id="ORD-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            commission_asset="USDT",
            timestamp=datetime.now(),
        )
        assert trade.trade_id == "TRD-001"
        assert trade.price == 50000.0

    def test_value(self):
        """测试成交金额"""
        trade = Trade(
            trade_id="TRD-001",
            order_id="ORD-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            commission_asset="USDT",
            timestamp=datetime.now(),
        )
        assert trade.value == 5000.0  # 50000 * 0.1

    def test_to_dict(self):
        """测试转字典"""
        now = datetime.now()
        trade = Trade(
            trade_id="TRD-001",
            order_id="ORD-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            price=50000.0,
            quantity=0.1,
            commission=5.0,
            commission_asset="USDT",
            timestamp=now,
            is_maker=True,
        )
        d = trade.to_dict()
        assert d['trade_id'] == "TRD-001"
        assert d['is_maker'] == True
        assert 'value' in d


class TestTicker:
    """行情快照数据类测试"""

    def test_creation(self):
        """测试创建行情快照"""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=49990.0,
            ask=50010.0,
            last=50000.0,
            volume_24h=1000000.0,
            change_24h=500.0,
            change_pct_24h=1.01,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now(),
        )
        assert ticker.symbol == "BTC/USDT"
        assert ticker.last == 50000.0

    def test_mid_price(self):
        """测试中间价"""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=49990.0,
            ask=50010.0,
            last=50000.0,
            volume_24h=1000000.0,
            change_24h=500.0,
            change_pct_24h=1.01,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now(),
        )
        assert ticker.mid_price == 50000.0

    def test_spread(self):
        """测试价差"""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=49990.0,
            ask=50010.0,
            last=50000.0,
            volume_24h=1000000.0,
            change_24h=500.0,
            change_pct_24h=1.01,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now(),
        )
        assert ticker.spread == 20.0

    def test_to_dict(self):
        """测试转字典"""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=49990.0,
            ask=50010.0,
            last=50000.0,
            volume_24h=1000000.0,
            change_24h=500.0,
            change_pct_24h=1.01,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now(),
        )
        d = ticker.to_dict()
        assert d['symbol'] == "BTC/USDT"
        assert 'mid_price' in d
        assert 'spread' in d


# ==================== Config Tests ====================

class TestExchangeConfig:
    """交易所配置测试"""

    def test_defaults(self):
        """测试默认配置"""
        config = ExchangeConfig()
        assert config.testnet == False
        assert config.timeout == 30.0
        assert config.rate_limit_per_second == 10.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = ExchangeConfig(
            name="binance",
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            timeout=60.0,
        )
        assert config.name == "binance"
        assert config.api_key == "test_key"
        assert config.testnet == True
        assert config.timeout == 60.0


# ==================== Exception Tests ====================

class TestExchangeError:
    """交易所错误测试"""

    def test_creation(self):
        """测试创建错误"""
        error = ExchangeError("Connection failed", code="E001")
        assert str(error) == "Connection failed"
        assert error.code == "E001"

    def test_without_code(self):
        """测试不带错误码"""
        error = ExchangeError("Unknown error")
        assert error.code is None


class TestOrderError:
    """订单错误测试"""

    def test_inheritance(self):
        """测试继承"""
        error = OrderError("Invalid order")
        assert isinstance(error, ExchangeError)


class TestInsufficientFundsError:
    """资金不足错误测试"""

    def test_inheritance(self):
        """测试继承"""
        error = InsufficientFundsError("Not enough balance")
        assert isinstance(error, OrderError)
        assert isinstance(error, ExchangeError)


class TestRateLimitError:
    """速率限制错误测试"""

    def test_creation(self):
        """测试创建"""
        error = RateLimitError("Too many requests", retry_after=60.0)
        assert str(error) == "Too many requests"
        assert error.retry_after == 60.0
