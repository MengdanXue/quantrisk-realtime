"""
Binance 适配器测试

测试内容:
- BinanceConfig 配置
- BinanceAdapter (现货)
- BinanceFuturesAdapter (期货)
- 签名生成
- API调用模拟
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json
import hmac
import hashlib

from quantrisk_realtime.exchanges import (
    BinanceAdapter,
    BinanceConfig,
    BinanceFuturesAdapter,
    create_binance_spot,
    create_binance_futures,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    Position,
    Balance,
    AccountInfo,
    Ticker,
    ExchangeError,
    OrderError,
)
from quantrisk_realtime.exchanges.base import PositionSide


# ==================== Config Tests ====================

class TestBinanceConfig:
    """Binance配置测试"""

    def test_defaults(self):
        """测试默认配置"""
        config = BinanceConfig()
        assert config.name == "binance"
        assert config.testnet == False
        assert config.recv_window == 5000

    def test_spot_urls(self):
        """测试现货URL"""
        config = BinanceConfig(testnet=False)
        assert config.spot_base_url == "https://api.binance.com"
        assert config.spot_testnet_url == "https://testnet.binance.vision"

    def test_testnet_urls(self):
        """测试测试网URL"""
        config = BinanceConfig(testnet=True)
        assert config.testnet == True

    def test_futures_urls(self):
        """测试期货URL"""
        config = BinanceConfig()
        assert "fapi" in config.futures_base_url


class TestCreateBinanceSpot:
    """测试创建现货适配器"""

    def test_create_without_credentials(self):
        """测试不带凭据创建"""
        adapter = create_binance_spot()
        assert adapter.name == "binance_spot"

    def test_create_with_credentials(self):
        """测试带凭据创建"""
        adapter = create_binance_spot(
            api_key="test_key",
            api_secret="test_secret",
        )
        assert adapter._config.api_key == "test_key"
        assert adapter._config.api_secret == "test_secret"

    def test_create_testnet(self):
        """测试创建测试网"""
        adapter = create_binance_spot(testnet=True)
        assert adapter._config.testnet == True


class TestCreateBinanceFutures:
    """测试创建期货适配器"""

    def test_create_without_credentials(self):
        """测试不带凭据创建"""
        adapter = create_binance_futures()
        assert adapter.name == "binance_futures"

    def test_create_with_credentials(self):
        """测试带凭据创建"""
        adapter = create_binance_futures(
            api_key="test_key",
            api_secret="test_secret",
        )
        assert adapter._config.api_key == "test_key"

    def test_create_testnet(self):
        """测试创建测试网"""
        adapter = create_binance_futures(testnet=True)
        assert adapter._config.testnet == True


# ==================== BinanceAdapter Tests ====================

class TestBinanceAdapter:
    """Binance现货适配器测试"""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return create_binance_spot(
            api_key="test_api_key",
            api_secret="test_api_secret",
        )

    def test_name(self, adapter):
        """测试名称"""
        assert adapter.name == "binance_spot"

    def test_is_connected_initial(self, adapter):
        """测试初始连接状态"""
        assert not adapter.is_connected

    def test_base_url(self, adapter):
        """测试基础URL"""
        assert adapter.base_url == "https://api.binance.com"

    def test_base_url_testnet(self):
        """测试测试网URL"""
        adapter = create_binance_spot(testnet=True)
        assert adapter.base_url == "https://testnet.binance.vision"

    def test_sign(self, adapter):
        """测试签名方法"""
        params = {
            "symbol": "BTCUSDT",
            "timestamp": 1234567890000,
        }
        signature = adapter._sign(params)

        # 手动计算签名验证
        query_string = "symbol=BTCUSDT&timestamp=1234567890000"
        expected_signature = hmac.new(
            b"test_api_secret",
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

        assert signature == expected_signature

    @pytest.mark.asyncio
    async def test_connect_success(self, adapter):
        """测试连接成功"""
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = {}
            result = await adapter.connect()
            assert result == True
            assert adapter.is_connected

    @pytest.mark.asyncio
    async def test_connect_failure(self, adapter):
        """测试连接失败"""
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Connection failed")
            result = await adapter.connect()
            assert result == False
            assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        """测试断开连接"""
        adapter._connected = True
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter):
        """测试获取账户信息"""
        mock_response = {
            "accountType": "SPOT",
            "balances": [
                {"asset": "USDT", "free": "10000.0", "locked": "500.0"},
                {"asset": "BTC", "free": "0.5", "locked": "0.0"},
            ],
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            account = await adapter.get_account_info()

            assert isinstance(account, AccountInfo)
            assert len(account.balances) == 2
            assert account.can_trade == True

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """测试获取余额"""
        mock_response = {
            "accountType": "SPOT",
            "balances": [
                {"asset": "USDT", "free": "10000.0", "locked": "500.0"},
                {"asset": "BTC", "free": "0.5", "locked": "0.0"},
            ],
            "canTrade": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            balance = await adapter.get_balance("USDT")

            assert balance is not None
            assert balance.asset == "USDT"
            assert balance.free == 10000.0
            assert balance.locked == 500.0

    @pytest.mark.asyncio
    async def test_get_balance_not_found(self, adapter):
        """测试获取不存在的余额"""
        mock_response = {
            "accountType": "SPOT",
            "balances": [
                {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
            ],
            "canTrade": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            balance = await adapter.get_balance("ETH")
            assert balance is None

    @pytest.mark.asyncio
    async def test_get_positions_spot(self, adapter):
        """测试现货获取持仓（返回空）"""
        positions = await adapter.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_create_market_order(self, adapter):
        """测试创建市价单"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "price": "0",
            "avgPrice": "50000.0",
            "time": 1234567890000,
            "updateTime": 1234567890000,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            )

            assert isinstance(order, Order)
            assert order.order_id == "12345"
            assert order.status == OrderStatus.FILLED
            assert order.filled_quantity == 0.1

    @pytest.mark.asyncio
    async def test_create_limit_order(self, adapter):
        """测试创建限价单"""
        mock_response = {
            "orderId": 12346,
            "clientOrderId": "CLI-002",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.0",
            "status": "NEW",
            "price": "55000.0",
            "avgPrice": "0",
            "time": 1234567890000,
            "timeInForce": "GTC",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=55000.0,
            )

            assert order.order_id == "12346"
            assert order.status == OrderStatus.NEW
            assert order.price == 55000.0

    @pytest.mark.asyncio
    async def test_cancel_order(self, adapter):
        """测试取消订单"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.0",
            "status": "CANCELED",
            "price": "50000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.cancel_order("BTCUSDT", "12345")

            assert order.status == OrderStatus.CANCELED

    @pytest.mark.asyncio
    async def test_get_order(self, adapter):
        """测试获取订单"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.05",
            "status": "PARTIALLY_FILLED",
            "price": "50000.0",
            "avgPrice": "50000.0",
            "time": 1234567890000,
            "updateTime": 1234567890000,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.get_order("BTCUSDT", "12345")

            assert order.order_id == "12345"
            assert order.status == OrderStatus.PARTIALLY_FILLED
            assert order.filled_quantity == 0.05

    @pytest.mark.asyncio
    async def test_get_open_orders(self, adapter):
        """测试获取未完成订单"""
        mock_response = [
            {
                "orderId": 12345,
                "clientOrderId": "CLI-001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "0.1",
                "executedQty": "0.0",
                "status": "NEW",
                "price": "50000.0",
            },
            {
                "orderId": 12346,
                "clientOrderId": "CLI-002",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": "1.0",
                "executedQty": "0.0",
                "status": "NEW",
                "price": "3000.0",
            },
        ]

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            orders = await adapter.get_open_orders()

            assert len(orders) == 2
            assert all(isinstance(o, Order) for o in orders)

    @pytest.mark.asyncio
    async def test_get_ticker(self, adapter):
        """测试获取行情"""
        mock_response = {
            "symbol": "BTCUSDT",
            "bidPrice": "49990.0",
            "askPrice": "50010.0",
            "lastPrice": "50000.0",
            "volume": "1000000.0",
            "priceChange": "500.0",
            "priceChangePercent": "1.01",
            "highPrice": "51000.0",
            "lowPrice": "49000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            ticker = await adapter.get_ticker("BTCUSDT")

            assert isinstance(ticker, Ticker)
            assert ticker.symbol == "BTCUSDT"
            assert ticker.last == 50000.0
            assert ticker.bid == 49990.0
            assert ticker.ask == 50010.0

    @pytest.mark.asyncio
    async def test_parse_order(self, adapter):
        """测试解析订单"""
        data = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.05",
            "status": "PARTIALLY_FILLED",
            "price": "50000.0",
            "time": 1234567890000,
        }

        order = adapter._parse_order(data)

        assert order.order_id == "12345"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 0.1
        assert order.filled_quantity == 0.05
        assert order.status == OrderStatus.PARTIALLY_FILLED


# ==================== BinanceFuturesAdapter Tests ====================

class TestBinanceFuturesAdapter:
    """Binance期货适配器测试"""

    @pytest.fixture
    def adapter(self):
        """创建期货适配器实例"""
        return create_binance_futures(
            api_key="test_api_key",
            api_secret="test_api_secret",
        )

    def test_name(self, adapter):
        """测试名称"""
        assert adapter.name == "binance_futures"

    def test_base_url(self, adapter):
        """测试期货基础URL"""
        assert adapter.base_url == "https://fapi.binance.com"

    def test_base_url_testnet(self):
        """测试期货测试网URL"""
        adapter = create_binance_futures(testnet=True)
        assert "testnet" in adapter.base_url

    @pytest.mark.asyncio
    async def test_get_account_info_futures(self, adapter):
        """测试期货账户信息"""
        mock_response = {
            "assets": [
                {"asset": "USDT", "balance": "10000.0", "availableBalance": "9000.0"},
            ],
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.1",
                    "entryPrice": "50000.0",
                    "markPrice": "52000.0",
                    "unrealizedProfit": "200.0",
                    "liquidationPrice": "40000.0",
                    "leverage": "10",
                    "marginType": "cross",
                    "positionSide": "LONG",
                    "initialMargin": "500.0",
                    "isolated": False,
                },
            ],
            "totalWalletBalance": "10000.0",
            "totalUnrealizedProfit": "200.0",
            "totalMarginBalance": "10200.0",
            "availableBalance": "9000.0",
            "maxWithdrawAmount": "8000.0",
            "canTrade": True,
            "canDeposit": True,
            "canWithdraw": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            account = await adapter.get_account_info()

            assert isinstance(account, AccountInfo)
            assert len(account.balances) == 1
            assert len(account.positions) == 1
            assert account.total_wallet_balance == 10000.0

    @pytest.mark.asyncio
    async def test_get_positions(self, adapter):
        """测试获取持仓"""
        mock_response = {
            "assets": [],
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.1",
                    "entryPrice": "50000.0",
                    "markPrice": "52000.0",
                    "unrealizedProfit": "200.0",
                    "liquidationPrice": "40000.0",
                    "leverage": "10",
                    "marginType": "cross",
                    "positionSide": "LONG",
                    "initialMargin": "500.0",
                },
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "-2.0",
                    "entryPrice": "3000.0",
                    "markPrice": "2900.0",
                    "unrealizedProfit": "200.0",
                    "liquidationPrice": "3500.0",
                    "leverage": "5",
                    "marginType": "cross",
                    "positionSide": "SHORT",
                    "initialMargin": "1000.0",
                },
            ],
            "totalWalletBalance": "10000.0",
            "totalUnrealizedProfit": "400.0",
            "totalMarginBalance": "10400.0",
            "availableBalance": "9000.0",
            "maxWithdrawAmount": "8000.0",
            "canTrade": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            positions = await adapter.get_positions()

            assert len(positions) == 2
            assert all(isinstance(p, Position) for p in positions)

            btc_pos = next(p for p in positions if p.symbol == "BTCUSDT")
            assert btc_pos.quantity == 0.1
            assert btc_pos.entry_price == 50000.0
            assert btc_pos.side == PositionSide.LONG

    @pytest.mark.asyncio
    async def test_get_positions_with_symbol(self, adapter):
        """测试获取指定交易对持仓"""
        mock_response = {
            "assets": [],
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.1",
                    "entryPrice": "50000.0",
                    "markPrice": "52000.0",
                    "unrealizedProfit": "200.0",
                    "leverage": "10",
                    "initialMargin": "500.0",
                },
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "1.0",
                    "entryPrice": "3000.0",
                    "markPrice": "3100.0",
                    "unrealizedProfit": "100.0",
                    "leverage": "5",
                    "initialMargin": "600.0",
                },
            ],
            "totalWalletBalance": "10000.0",
            "canTrade": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            positions = await adapter.get_positions("BTCUSDT")

            assert len(positions) == 1
            assert positions[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_create_futures_order(self, adapter):
        """测试创建期货订单"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "avgPrice": "50000.0",
            "updateTime": 1234567890000,
            "positionSide": "LONG",
            "reduceOnly": False,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            )

            assert order.order_id == "12345"
            assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_create_futures_reduce_only_order(self, adapter):
        """测试创建平仓订单"""
        mock_response = {
            "orderId": 12346,
            "clientOrderId": "CLI-002",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "avgPrice": "52000.0",
            "updateTime": 1234567890000,
            "positionSide": "LONG",
            "reduceOnly": True,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.1,
                reduce_only=True,
            )

            assert order.reduce_only == True

    @pytest.mark.asyncio
    async def test_set_leverage(self, adapter):
        """测试设置杠杆"""
        mock_response = {
            "leverage": 20,
            "maxNotionalValue": "10000000",
            "symbol": "BTCUSDT",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await adapter.set_leverage("BTCUSDT", 20)

            assert result['leverage'] == 20

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, adapter):
        """测试获取资金费率"""
        mock_response = {
            "symbol": "BTCUSDT",
            "markPrice": "50000.0",
            "indexPrice": "49990.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1234567890000,
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await adapter.get_funding_rate("BTCUSDT")

            assert result['symbol'] == "BTCUSDT"
            assert result['funding_rate'] == 0.0001


# ==================== Convenience Method Tests ====================

class TestConvenienceMethods:
    """便捷方法测试"""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return create_binance_spot(
            api_key="test_api_key",
            api_secret="test_api_secret",
        )

    @pytest.mark.asyncio
    async def test_market_buy(self, adapter):
        """测试市价买入"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "avgPrice": "50000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.market_buy("BTCUSDT", 0.1)

            assert order.side == OrderSide.BUY
            assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_market_sell(self, adapter):
        """测试市价卖出"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "avgPrice": "50000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.market_sell("BTCUSDT", 0.1)

            assert order.side == OrderSide.SELL
            assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_limit_buy(self, adapter):
        """测试限价买入"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.0",
            "status": "NEW",
            "price": "48000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.limit_buy("BTCUSDT", 0.1, 48000.0)

            assert order.side == OrderSide.BUY
            assert order.order_type == OrderType.LIMIT
            assert order.price == 48000.0

    @pytest.mark.asyncio
    async def test_limit_sell(self, adapter):
        """测试限价卖出"""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.0",
            "status": "NEW",
            "price": "55000.0",
        }

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            order = await adapter.limit_sell("BTCUSDT", 0.1, 55000.0)

            assert order.side == OrderSide.SELL
            assert order.order_type == OrderType.LIMIT


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """错误处理测试"""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return create_binance_spot(
            api_key="test_api_key",
            api_secret="test_api_secret",
        )

    @pytest.mark.asyncio
    async def test_order_error(self, adapter):
        """测试订单错误"""
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.side_effect = OrderError("Insufficient balance", code="-2010")

            with pytest.raises(OrderError) as exc_info:
                await adapter.create_order(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=1000.0,
                )

            assert "-2010" in str(exc_info.value.code)

    @pytest.mark.asyncio
    async def test_exchange_error(self, adapter):
        """测试交易所错误"""
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.side_effect = ExchangeError("API Error", code="-1000")

            with pytest.raises(ExchangeError):
                await adapter.get_ticker("BTCUSDT")


# ==================== Order Parsing Tests ====================

class TestOrderParsing:
    """订单解析测试"""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return create_binance_spot()

    def test_parse_order_basic(self, adapter):
        """测试基本订单解析"""
        data = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.1",
            "executedQty": "0.1",
            "status": "FILLED",
            "price": "0",
            "avgPrice": "50000.0",
        }

        order = adapter._parse_order(data)

        assert order.order_id == "12345"
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED

    def test_parse_order_status_new(self, adapter):
        """测试NEW状态解析"""
        data = {
            "orderId": 12345,
            "clientOrderId": "CLI-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.1",
            "executedQty": "0.0",
            "status": "NEW",
            "price": "50000.0",
        }

        order = adapter._parse_order(data)
        assert order.status == OrderStatus.NEW

    def test_parse_order_status_partially_filled(self, adapter):
        """测试PARTIALLY_FILLED状态解析"""
        data = {
            "orderId": 12345,
            "status": "PARTIALLY_FILLED",
            "origQty": "1.0",
            "executedQty": "0.5",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": "50000.0",
        }

        order = adapter._parse_order(data)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 0.5

    def test_parse_order_status_canceled(self, adapter):
        """测试CANCELED状态解析"""
        data = {
            "orderId": 12345,
            "status": "CANCELED",
            "origQty": "0.1",
            "executedQty": "0.0",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": "50000.0",
        }

        order = adapter._parse_order(data)
        assert order.status == OrderStatus.CANCELED


# ==================== Integration Style Tests ====================

class TestIntegrationStyle:
    """集成风格测试（不需要真实API）"""

    @pytest.mark.asyncio
    async def test_full_trading_flow(self):
        """测试完整交易流程"""
        adapter = create_binance_spot(
            api_key="test_key",
            api_secret="test_secret",
        )

        # 模拟连接
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}
            connected = await adapter.connect()
            assert connected

        # 模拟获取账户信息
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "accountType": "SPOT",
                "balances": [
                    {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
                ],
                "canTrade": True,
            }
            account = await adapter.get_account_info()
            assert account.get_balance("USDT").free == 10000.0

        # 模拟获取行情
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "symbol": "BTCUSDT",
                "bidPrice": "49990.0",
                "askPrice": "50010.0",
                "lastPrice": "50000.0",
                "volume": "1000000.0",
                "priceChange": "500.0",
                "priceChangePercent": "1.01",
                "highPrice": "51000.0",
                "lowPrice": "49000.0",
            }
            ticker = await adapter.get_ticker("BTCUSDT")
            assert ticker.last == 50000.0

        # 模拟下单
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "orderId": 12345,
                "clientOrderId": "CLI-001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "0.1",
                "executedQty": "0.0",
                "status": "NEW",
                "price": "48000.0",
            }
            order = await adapter.limit_buy("BTCUSDT", 0.1, 48000.0)
            assert order.status == OrderStatus.NEW

        # 模拟取消订单
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "orderId": 12345,
                "clientOrderId": "CLI-001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "0.1",
                "executedQty": "0.0",
                "status": "CANCELED",
                "price": "48000.0",
            }
            canceled = await adapter.cancel_order("BTCUSDT", "12345")
            assert canceled.status == OrderStatus.CANCELED

        # 断开连接
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_futures_position_management(self):
        """测试期货持仓管理"""
        adapter = create_binance_futures(
            api_key="test_key",
            api_secret="test_secret",
        )

        # 模拟开仓
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = {
                "orderId": 12345,
                "clientOrderId": "CLI-001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "origQty": "0.1",
                "executedQty": "0.1",
                "status": "FILLED",
                "avgPrice": "50000.0",
                "positionSide": "LONG",
            }
            order = await adapter.market_buy("BTCUSDT", 0.1)
            assert order.status == OrderStatus.FILLED

        # 模拟查询持仓
        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = {
                "assets": [],
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "positionAmt": "0.1",
                        "entryPrice": "50000.0",
                        "markPrice": "52000.0",
                        "unrealizedProfit": "200.0",
                        "liquidationPrice": "40000.0",
                        "leverage": "10",
                        "marginType": "cross",
                        "positionSide": "LONG",
                        "initialMargin": "500.0",
                    }
                ],
                "totalWalletBalance": "10000.0",
                "canTrade": True,
            }
            positions = await adapter.get_positions("BTCUSDT")
            assert len(positions) == 1
            assert positions[0].unrealized_pnl == 200.0

        # 模拟平仓
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=52000.0,
        )

        with patch.object(adapter, '_request', new_callable=AsyncMock) as mock:
            mock.return_value = {
                "orderId": 12346,
                "clientOrderId": "CLI-002",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "MARKET",
                "origQty": "0.1",
                "executedQty": "0.1",
                "status": "FILLED",
                "avgPrice": "52000.0",
                "reduceOnly": True,
            }
            close_order = await adapter.close_position("BTCUSDT", position)
            assert close_order.status == OrderStatus.FILLED
