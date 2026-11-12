"""
Binance WebSocket 适配器测试

测试内容:
- 数据类型
- 配置类
- WebSocket客户端
- 数据解析
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.exchanges.binance_websocket import (
    # Enums
    StreamType,
    KlineInterval,
    ConnectionState,
    # Data classes
    TickerData,
    TradeData,
    KlineData,
    DepthLevel,
    DepthData,
    BookTickerData,
    MarkPriceData,
    UserBalanceUpdate,
    UserPositionUpdate,
    UserOrderUpdate,
    # Config
    WebSocketConfig,
    # Client
    BinanceWebSocket,
    # Factory
    create_spot_websocket,
    create_futures_websocket,
)


# ==================== Enum Tests ====================

class TestStreamType:
    """数据流类型测试"""

    def test_all_values(self):
        """测试所有值"""
        assert StreamType.TICKER == "ticker"
        assert StreamType.TRADE == "trade"
        assert StreamType.KLINE == "kline"
        assert StreamType.DEPTH == "depth"
        assert StreamType.BOOK_TICKER == "bookTicker"
        assert StreamType.MARK_PRICE == "markPrice"
        assert StreamType.USER_DATA == "userData"


class TestKlineInterval:
    """K线间隔测试"""

    def test_minutes(self):
        """测试分钟级别"""
        assert KlineInterval.M1 == "1m"
        assert KlineInterval.M5 == "5m"
        assert KlineInterval.M15 == "15m"
        assert KlineInterval.M30 == "30m"

    def test_hours(self):
        """测试小时级别"""
        assert KlineInterval.H1 == "1h"
        assert KlineInterval.H4 == "4h"
        assert KlineInterval.H12 == "12h"

    def test_days(self):
        """测试日级别"""
        assert KlineInterval.D1 == "1d"
        assert KlineInterval.W1 == "1w"
        assert KlineInterval.MO1 == "1M"


class TestConnectionState:
    """连接状态测试"""

    def test_all_states(self):
        """测试所有状态"""
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.CLOSING == "closing"
        assert ConnectionState.CLOSED == "closed"


# ==================== Data Class Tests ====================

class TestTickerData:
    """行情数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'c': '50000.00',
            'p': '500.00',
            'P': '1.01',
            'h': '51000.00',
            'l': '49000.00',
            'v': '10000.00',
            'q': '500000000.00',
            'o': '49500.00',
            'E': 1704067200000,
        }

        ticker = TickerData.from_binance(data)

        assert ticker.symbol == 'BTCUSDT'
        assert ticker.price == 50000.00
        assert ticker.price_change == 500.00
        assert ticker.price_change_pct == 1.01
        assert ticker.high_24h == 51000.00
        assert ticker.low_24h == 49000.00
        assert ticker.volume_24h == 10000.00
        assert ticker.quote_volume_24h == 500000000.00
        assert ticker.open_price == 49500.00

    def test_missing_fields(self):
        """测试缺少字段"""
        data = {'s': 'BTCUSDT'}
        ticker = TickerData.from_binance(data)
        assert ticker.symbol == 'BTCUSDT'
        assert ticker.price == 0


class TestTradeData:
    """成交数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            't': 123456789,
            'p': '50000.00',
            'q': '0.1',
            'm': True,
            'T': 1704067200000,
        }

        trade = TradeData.from_binance(data)

        assert trade.symbol == 'BTCUSDT'
        assert trade.trade_id == 123456789
        assert trade.price == 50000.00
        assert trade.quantity == 0.1
        assert trade.buyer_maker == True


class TestKlineData:
    """K线数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'k': {
                'i': '1m',
                't': 1704067200000,
                'T': 1704067259999,
                'o': '50000.00',
                'h': '50100.00',
                'l': '49900.00',
                'c': '50050.00',
                'v': '100.0',
                'q': '5000000.00',
                'n': 1000,
                'x': True,
            }
        }

        kline = KlineData.from_binance(data)

        assert kline.symbol == 'BTCUSDT'
        assert kline.interval == '1m'
        assert kline.open == 50000.00
        assert kline.high == 50100.00
        assert kline.low == 49900.00
        assert kline.close == 50050.00
        assert kline.volume == 100.0
        assert kline.trades == 1000
        assert kline.is_closed == True


class TestDepthLevel:
    """深度层级测试"""

    def test_from_list(self):
        """测试从列表创建"""
        level = DepthLevel.from_list(['50000.00', '10.0'])
        assert level.price == 50000.00
        assert level.quantity == 10.0


class TestDepthData:
    """深度数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'b': [['49900.00', '5.0'], ['49800.00', '10.0']],
            'a': [['50000.00', '3.0'], ['50100.00', '8.0']],
            'u': 123456789,
            'E': 1704067200000,
        }

        depth = DepthData.from_binance(data)

        assert depth.symbol == 'BTCUSDT'
        assert len(depth.bids) == 2
        assert len(depth.asks) == 2
        assert depth.bids[0].price == 49900.00
        assert depth.bids[0].quantity == 5.0
        assert depth.asks[0].price == 50000.00
        assert depth.asks[0].quantity == 3.0
        assert depth.last_update_id == 123456789


class TestBookTickerData:
    """最优挂单数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'b': '49999.00',
            'B': '5.0',
            'a': '50001.00',
            'A': '3.0',
        }

        book = BookTickerData.from_binance(data)

        assert book.symbol == 'BTCUSDT'
        assert book.bid_price == 49999.00
        assert book.bid_qty == 5.0
        assert book.ask_price == 50001.00
        assert book.ask_qty == 3.0


class TestMarkPriceData:
    """标记价格数据测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'p': '50000.00',
            'i': '49990.00',
            'r': '0.0001',
            'T': 1704067200000,
            'E': 1704067100000,
        }

        mark = MarkPriceData.from_binance(data)

        assert mark.symbol == 'BTCUSDT'
        assert mark.mark_price == 50000.00
        assert mark.index_price == 49990.00
        assert mark.funding_rate == 0.0001


class TestUserBalanceUpdate:
    """用户余额更新测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            'a': 'USDT',
            'wb': '10000.00',
            'cw': '9500.00',
            'bc': '100.00',
        }

        balance = UserBalanceUpdate.from_binance(data)

        assert balance.asset == 'USDT'
        assert balance.wallet_balance == 10000.00
        assert balance.cross_wallet_balance == 9500.00
        assert balance.balance_change == 100.00


class TestUserPositionUpdate:
    """用户持仓更新测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'pa': '0.5',
            'ep': '50000.00',
            'up': '100.00',
            'mt': 'cross',
            'iw': '0',
            'ps': 'LONG',
        }

        position = UserPositionUpdate.from_binance(data)

        assert position.symbol == 'BTCUSDT'
        assert position.position_amount == 0.5
        assert position.entry_price == 50000.00
        assert position.unrealized_pnl == 100.00
        assert position.margin_type == 'cross'
        assert position.position_side == 'LONG'


class TestUserOrderUpdate:
    """用户订单更新测试"""

    def test_from_binance(self):
        """测试从Binance数据创建"""
        data = {
            's': 'BTCUSDT',
            'i': 123456789,
            'c': 'client_order_id',
            'S': 'BUY',
            'o': 'LIMIT',
            'f': 'GTC',
            'q': '0.1',
            'p': '50000.00',
            'sp': '0',
            'X': 'NEW',
            'z': '0',
            'ap': '0',
            'n': '0',
            'N': 'USDT',
            'T': 1704067200000,
        }

        order = UserOrderUpdate.from_binance(data)

        assert order.symbol == 'BTCUSDT'
        assert order.order_id == 123456789
        assert order.client_order_id == 'client_order_id'
        assert order.side == 'BUY'
        assert order.order_type == 'LIMIT'
        assert order.quantity == 0.1
        assert order.price == 50000.00
        assert order.status == 'NEW'


# ==================== Config Tests ====================

class TestWebSocketConfig:
    """WebSocket配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = WebSocketConfig()

        assert config.spot_stream_url == "wss://stream.binance.com:9443/ws"
        assert config.futures_stream_url == "wss://fstream.binance.com/ws"
        assert config.ping_interval == 30.0
        assert config.reconnect_delay == 5.0
        assert config.max_reconnect_attempts == 10
        assert config.testnet == False
        assert config.is_futures == False

    def test_custom_values(self):
        """测试自定义值"""
        config = WebSocketConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            is_futures=True,
            ping_interval=60.0,
        )

        assert config.api_key == "test_key"
        assert config.testnet == True
        assert config.is_futures == True
        assert config.ping_interval == 60.0


# ==================== WebSocket Client Tests ====================

class TestBinanceWebSocket:
    """WebSocket客户端测试"""

    def test_init(self):
        """测试初始化"""
        ws = BinanceWebSocket()

        assert ws.state == ConnectionState.DISCONNECTED
        assert ws.is_connected == False
        assert len(ws.subscriptions) == 0

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = WebSocketConfig(testnet=True, is_futures=True)
        ws = BinanceWebSocket(config=config)

        assert ws.config.testnet == True
        assert ws.config.is_futures == True

    def test_get_base_url_spot(self):
        """测试获取现货URL"""
        ws = BinanceWebSocket()
        assert "stream.binance.com" in ws._get_base_url()

    def test_get_base_url_futures(self):
        """测试获取合约URL"""
        config = WebSocketConfig(is_futures=True)
        ws = BinanceWebSocket(config=config)
        assert "fstream.binance.com" in ws._get_base_url()

    def test_get_base_url_testnet(self):
        """测试获取测试网URL"""
        config = WebSocketConfig(testnet=True)
        ws = BinanceWebSocket(config=config)
        assert "testnet" in ws._get_base_url()

    def test_callback_registration(self):
        """测试回调注册"""
        ws = BinanceWebSocket()

        async def ticker_callback(data):
            pass

        async def trade_callback(data):
            pass

        ws.on_ticker(ticker_callback)
        ws.on_trade(trade_callback)

        assert len(ws._ticker_callbacks) == 1
        assert len(ws._trade_callbacks) == 1


class TestBinanceWebSocketSubscriptions:
    """WebSocket订阅测试"""

    @pytest.fixture
    def ws(self):
        """创建WebSocket客户端"""
        return BinanceWebSocket()

    @pytest.mark.asyncio
    async def test_subscribe_ticker(self, ws):
        """测试订阅行情"""
        await ws.subscribe_ticker('BTCUSDT')
        assert 'btcusdt@ticker' in ws.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_trade(self, ws):
        """测试订阅成交"""
        await ws.subscribe_trade('BTCUSDT')
        assert 'btcusdt@trade' in ws.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_kline(self, ws):
        """测试订阅K线"""
        await ws.subscribe_kline('BTCUSDT', KlineInterval.M1)
        assert 'btcusdt@kline_1m' in ws.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_depth(self, ws):
        """测试订阅深度"""
        await ws.subscribe_depth('BTCUSDT', 20)
        assert 'btcusdt@depth20' in ws.subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_book_ticker(self, ws):
        """测试订阅最优挂单"""
        await ws.subscribe_book_ticker('BTCUSDT')
        assert 'btcusdt@bookTicker' in ws.subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self, ws):
        """测试取消订阅"""
        await ws.subscribe_ticker('BTCUSDT')
        assert 'btcusdt@ticker' in ws.subscriptions

        await ws.unsubscribe('btcusdt@ticker')
        assert 'btcusdt@ticker' not in ws.subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, ws):
        """测试取消所有订阅"""
        await ws.subscribe_ticker('BTCUSDT')
        await ws.subscribe_trade('ETHUSDT')
        await ws.subscribe_kline('BTCUSDT', KlineInterval.H1)

        assert len(ws.subscriptions) == 3

        await ws.unsubscribe_all()
        assert len(ws.subscriptions) == 0


class TestMessageParsing:
    """消息解析测试"""

    @pytest.fixture
    def ws(self):
        """创建WebSocket客户端"""
        return BinanceWebSocket()

    @pytest.mark.asyncio
    async def test_handle_ticker_message(self, ws):
        """测试处理Ticker消息"""
        received = []

        async def callback(data):
            received.append(data)

        ws.on_ticker(callback)

        # 模拟消息
        import json
        msg = json.dumps({
            'e': '24hrTicker',
            's': 'BTCUSDT',
            'c': '50000.00',
            'p': '500.00',
            'P': '1.01',
            'h': '51000.00',
            'l': '49000.00',
            'v': '10000.00',
            'q': '500000000.00',
            'o': '49500.00',
        })

        await ws._handle_message(msg)

        assert len(received) == 1
        assert received[0].symbol == 'BTCUSDT'
        assert received[0].price == 50000.00

    @pytest.mark.asyncio
    async def test_handle_trade_message(self, ws):
        """测试处理Trade消息"""
        received = []

        async def callback(data):
            received.append(data)

        ws.on_trade(callback)

        import json
        msg = json.dumps({
            'e': 'trade',
            's': 'BTCUSDT',
            't': 123456789,
            'p': '50000.00',
            'q': '0.1',
            'm': True,
        })

        await ws._handle_message(msg)

        assert len(received) == 1
        assert received[0].symbol == 'BTCUSDT'
        assert received[0].trade_id == 123456789

    @pytest.mark.asyncio
    async def test_handle_kline_message(self, ws):
        """测试处理Kline消息"""
        received = []

        async def callback(data):
            received.append(data)

        ws.on_kline(callback)

        import json
        msg = json.dumps({
            'e': 'kline',
            's': 'BTCUSDT',
            'k': {
                'i': '1m',
                't': 1704067200000,
                'T': 1704067259999,
                'o': '50000.00',
                'h': '50100.00',
                'l': '49900.00',
                'c': '50050.00',
                'v': '100.0',
                'q': '5000000.00',
                'n': 1000,
                'x': True,
            }
        })

        await ws._handle_message(msg)

        assert len(received) == 1
        assert received[0].symbol == 'BTCUSDT'
        assert received[0].is_closed == True

    @pytest.mark.asyncio
    async def test_handle_combined_stream_message(self, ws):
        """测试处理组合流消息"""
        received = []

        async def callback(data):
            received.append(data)

        ws.on_ticker(callback)

        import json
        msg = json.dumps({
            'stream': 'btcusdt@ticker',
            'data': {
                'e': '24hrTicker',
                's': 'BTCUSDT',
                'c': '50000.00',
                'p': '500.00',
                'P': '1.01',
                'h': '51000.00',
                'l': '49000.00',
                'v': '10000.00',
                'q': '500000000.00',
                'o': '49500.00',
            }
        })

        await ws._handle_message(msg)

        assert len(received) == 1


# ==================== Factory Function Tests ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_spot_websocket(self):
        """测试创建现货WebSocket"""
        ws = create_spot_websocket()
        assert ws.config.is_futures == False

    def test_create_spot_websocket_testnet(self):
        """测试创建现货测试网WebSocket"""
        ws = create_spot_websocket(testnet=True)
        assert ws.config.testnet == True
        assert "testnet" in ws._get_base_url()

    def test_create_futures_websocket(self):
        """测试创建合约WebSocket"""
        ws = create_futures_websocket()
        assert ws.config.is_futures == True

    def test_create_futures_websocket_with_credentials(self):
        """测试创建带凭证的合约WebSocket"""
        ws = create_futures_websocket(
            api_key="test_key",
            api_secret="test_secret",
        )
        assert ws.config.api_key == "test_key"
        assert ws.config.api_secret == "test_secret"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
