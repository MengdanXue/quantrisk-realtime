"""
P&L推送引擎单元测试
PnL Streamer Unit Tests

测试覆盖:
- 推送器初始化和生命周期
- 订阅管理
- P&L计算
- 增量更新逻辑
- 心跳机制
- 统计信息

作者: TraderX Team
创建: 2025-12-24
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.pnl_streamer import (
    PnLStreamer,
    PortfolioPnL,
    PositionPnL,
    PnLUpdate,
    StreamerConfig,
    SubscriptionType,
    UpdateType,
    StreamerState,
    create_pnl_streamer,
)


# ============ Mock WebSocket ============

class MockWebSocket:
    """模拟WebSocket连接"""
    def __init__(self):
        self.messages = []
        self.closed = False

    async def send_text(self, message: str):
        if self.closed:
            raise Exception("Connection closed")
        self.messages.append(message)

    async def send(self, message: str):
        await self.send_text(message)

    def get_messages(self):
        return self.messages

    def close(self):
        self.closed = True


# ============ Fixtures ============

@pytest.fixture
def streamer_config():
    """推送器配置"""
    return StreamerConfig(
        update_interval_seconds=0.1,  # 快速测试
        heartbeat_interval_seconds=1.0,
        min_pnl_change_threshold=0.01,
    )


@pytest.fixture
def streamer(streamer_config):
    """创建推送器"""
    return PnLStreamer(config=streamer_config)


@pytest.fixture
def mock_websocket():
    """模拟WebSocket"""
    return MockWebSocket()


# ============ 基础测试 ============

class TestStreamerConfig:
    """测试配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = StreamerConfig()

        assert config.update_interval_seconds == 1.0
        assert config.heartbeat_interval_seconds == 30.0
        assert config.min_pnl_change_threshold == 0.01

    def test_custom_config(self):
        """测试自定义配置"""
        config = StreamerConfig(
            update_interval_seconds=0.5,
            max_subscriptions_per_client=100,
        )

        assert config.update_interval_seconds == 0.5
        assert config.max_subscriptions_per_client == 100


class TestStreamerInitialization:
    """测试初始化"""

    def test_basic_init(self):
        """测试基本初始化"""
        streamer = PnLStreamer()

        assert streamer.state == StreamerState.STOPPED
        assert streamer.subscriber_count == 0

    def test_init_with_config(self, streamer_config):
        """测试带配置初始化"""
        streamer = PnLStreamer(config=streamer_config)

        assert streamer.config.update_interval_seconds == 0.1

    def test_init_with_callbacks(self):
        """测试带回调初始化"""
        on_update = AsyncMock()
        on_error = AsyncMock()

        streamer = PnLStreamer(
            on_pnl_update=on_update,
            on_error=on_error,
        )

        assert streamer.on_pnl_update == on_update
        assert streamer.on_error == on_error


# ============ 生命周期测试 ============

class TestStreamerLifecycle:
    """测试生命周期"""

    @pytest.mark.asyncio
    async def test_start_stop(self, streamer):
        """测试启动和停止"""
        assert streamer.state == StreamerState.STOPPED

        await streamer.start()
        assert streamer.state == StreamerState.RUNNING
        assert streamer.is_running

        await streamer.stop()
        assert streamer.state == StreamerState.STOPPED
        assert not streamer.is_running

    @pytest.mark.asyncio
    async def test_pause_resume(self, streamer):
        """测试暂停和恢复"""
        await streamer.start()

        await streamer.pause()
        assert streamer.state == StreamerState.PAUSED

        await streamer.resume()
        assert streamer.state == StreamerState.RUNNING

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_double_start(self, streamer):
        """测试重复启动"""
        await streamer.start()
        await streamer.start()  # 不应报错

        assert streamer.state == StreamerState.RUNNING

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self, streamer):
        """测试重复停止"""
        await streamer.start()
        await streamer.stop()
        await streamer.stop()  # 不应报错

        assert streamer.state == StreamerState.STOPPED


# ============ 订阅管理测试 ============

class TestSubscriptionManagement:
    """测试订阅管理"""

    @pytest.mark.asyncio
    async def test_subscribe_portfolio(self, streamer, mock_websocket):
        """测试订阅投资组合"""
        await streamer.start()

        success = await streamer.subscribe_portfolio(mock_websocket, 1)

        assert success
        assert streamer.subscriber_count == 1

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_subscribe_position(self, streamer, mock_websocket):
        """测试订阅持仓"""
        await streamer.start()

        success = await streamer.subscribe_position(mock_websocket, 1001)

        assert success

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_subscribe_account(self, streamer, mock_websocket):
        """测试订阅账户"""
        await streamer.start()

        success = await streamer.subscribe_account(mock_websocket, "account-001")

        assert success

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_subscription_limit(self, streamer):
        """测试订阅限制"""
        streamer.config.max_subscriptions_per_client = 2

        await streamer.start()

        ws = MockWebSocket()

        # 前两个应该成功
        await streamer.subscribe_portfolio(ws, 1)
        await streamer.subscribe_portfolio(ws, 2)

        # 第三个应该失败
        success = await streamer.subscribe_portfolio(ws, 3)
        assert not success

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, streamer, mock_websocket):
        """测试取消订阅"""
        await streamer.start()

        await streamer.subscribe_portfolio(mock_websocket, 1)
        assert streamer.subscriber_count == 1

        await streamer.unsubscribe(mock_websocket, "portfolio:1")
        assert streamer.subscriber_count == 0

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, streamer, mock_websocket):
        """测试取消所有订阅"""
        await streamer.start()

        await streamer.subscribe_portfolio(mock_websocket, 1)
        await streamer.subscribe_portfolio(mock_websocket, 2)
        await streamer.subscribe_position(mock_websocket, 1001)

        await streamer.unsubscribe(mock_websocket)  # 取消所有

        assert streamer.subscriber_count == 0

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_handle_disconnect(self, streamer, mock_websocket):
        """测试处理断开连接"""
        await streamer.start()

        await streamer.subscribe_portfolio(mock_websocket, 1)
        await streamer.subscribe_portfolio(mock_websocket, 2)

        await streamer.handle_disconnect(mock_websocket)

        assert streamer.subscriber_count == 0

        await streamer.stop()


# ============ 数据结构测试 ============

class TestPositionPnL:
    """测试持仓P&L数据结构"""

    def test_basic_creation(self):
        """测试基本创建"""
        pnl = PositionPnL(
            position_id=1001,
            symbol="AAPL",
            current_price=150.0,
            quantity=100,
        )

        assert pnl.position_id == 1001
        assert pnl.symbol == "AAPL"

    def test_to_dict(self):
        """测试转换为字典"""
        pnl = PositionPnL(
            position_id=1001,
            symbol="AAPL",
            unrealized_pnl=500.0,
        )

        d = pnl.to_dict()

        assert d['position_id'] == 1001
        assert d['symbol'] == "AAPL"
        assert d['unrealized_pnl'] == 500.0


class TestPortfolioPnL:
    """测试投资组合P&L数据结构"""

    def test_basic_creation(self):
        """测试基本创建"""
        pnl = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1500.0,
        )

        assert pnl.portfolio_id == 1
        assert pnl.total_unrealized_pnl == 1500.0

    def test_to_dict_without_positions(self):
        """测试转换为字典(不含持仓)"""
        pnl = PortfolioPnL(portfolio_id=1)

        d = pnl.to_dict(include_positions=False)

        assert 'portfolio_id' in d
        assert 'positions' not in d

    def test_to_dict_with_positions(self):
        """测试转换为字典(含持仓)"""
        pnl = PortfolioPnL(portfolio_id=1)
        pnl.positions = [
            PositionPnL(position_id=1001, symbol="AAPL"),
        ]

        d = pnl.to_dict(include_positions=True)

        assert 'positions' in d
        assert len(d['positions']) == 1


class TestPnLUpdate:
    """测试P&L更新消息"""

    def test_to_json_full_update(self):
        """测试全量更新JSON"""
        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)
        update = PnLUpdate(
            update_type=UpdateType.FULL,
            subscription_type=SubscriptionType.PORTFOLIO,
            portfolio_pnl=pnl,
            sequence=1,
        )

        json_str = update.to_json()

        assert '"type": "pnl_update"' in json_str
        assert '"update_type": "full"' in json_str

    def test_to_json_incremental_update(self):
        """测试增量更新JSON"""
        pnl = PortfolioPnL(portfolio_id=1)
        update = PnLUpdate(
            update_type=UpdateType.INCREMENTAL,
            subscription_type=SubscriptionType.PORTFOLIO,
            portfolio_pnl=pnl,
            pnl_change=50.0,
            sequence=2,
        )

        json_str = update.to_json()

        assert '"pnl_change": 50.0' in json_str


# ============ P&L计算测试 ============

class TestPnLCalculation:
    """测试P&L计算"""

    @pytest.mark.asyncio
    async def test_mock_pnl_generation(self, streamer):
        """测试模拟P&L生成"""
        await streamer.start()

        pnl = await streamer._calculate_portfolio_pnl(1)

        assert pnl is not None
        assert pnl.portfolio_id == 1
        assert pnl.position_count > 0

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_should_send_update_no_cache(self, streamer):
        """测试无缓存时应该发送"""
        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)

        should_send = streamer._should_send_update(1, pnl)

        assert should_send  # 没有缓存，应该发送

    @pytest.mark.asyncio
    async def test_should_send_update_significant_change(self, streamer):
        """测试显著变化时应该发送"""
        # 设置缓存
        cached = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)
        streamer._portfolio_pnl_cache[1] = cached

        # 变化超过阈值
        new_pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1020.0)  # +2%

        should_send = streamer._should_send_update(1, new_pnl)

        assert should_send

    @pytest.mark.asyncio
    async def test_should_not_send_update_small_change(self, streamer):
        """测试小变化时不应该发送"""
        # 设置缓存
        cached = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1000.0,
            total_market_value=100000.0,
        )
        streamer._portfolio_pnl_cache[1] = cached

        # 变化很小
        new_pnl = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1001.0,  # 0.1%
            total_market_value=100010.0,  # 0.01%
        )

        should_send = streamer._should_send_update(1, new_pnl)

        assert not should_send


# ============ 推送测试 ============

class TestBroadcast:
    """测试广播"""

    @pytest.mark.asyncio
    async def test_initial_data_sent(self, streamer, mock_websocket):
        """测试发送初始数据"""
        await streamer.start()

        await streamer.subscribe_portfolio(mock_websocket, 1)

        # 等待一点时间让初始数据发送
        await asyncio.sleep(0.2)

        messages = mock_websocket.get_messages()
        assert len(messages) >= 1

        # 第一条消息应该是全量更新
        import json
        first_msg = json.loads(messages[0])
        assert first_msg['update_type'] == 'full'

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self, streamer):
        """测试广播到多个订阅者"""
        await streamer.start()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await streamer.subscribe_portfolio(ws1, 1)
        await streamer.subscribe_portfolio(ws2, 1)

        # 等待更新
        await asyncio.sleep(0.3)

        # 两个客户端都应该收到消息
        assert len(ws1.get_messages()) >= 1
        assert len(ws2.get_messages()) >= 1

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_dead_connection_cleanup(self, streamer):
        """测试断开连接清理"""
        await streamer.start()

        ws = MockWebSocket()
        await streamer.subscribe_portfolio(ws, 1)

        # 关闭连接
        ws.close()

        # 等待更新循环检测到断开
        await asyncio.sleep(0.3)

        # 订阅者应该被清理
        assert streamer.subscriber_count == 0

        await streamer.stop()


# ============ 统计测试 ============

class TestStatistics:
    """测试统计信息"""

    @pytest.mark.asyncio
    async def test_get_statistics(self, streamer, mock_websocket):
        """测试获取统计信息"""
        await streamer.start()
        await streamer.subscribe_portfolio(mock_websocket, 1)

        stats = streamer.get_statistics()

        assert stats['state'] == 'running'
        assert stats['subscriber_count'] == 1
        assert stats['portfolio_subscriptions'] == 1
        assert 'uptime_seconds' in stats

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_update_count_increments(self, streamer, mock_websocket):
        """测试更新计数"""
        await streamer.start()
        await streamer.subscribe_portfolio(mock_websocket, 1)

        # 等待几次更新
        await asyncio.sleep(0.5)

        stats = streamer.get_statistics()
        assert stats['update_count'] > 0

        await streamer.stop()


# ============ 工厂函数测试 ============

class TestFactoryFunction:
    """测试工厂函数"""

    def test_create_pnl_streamer(self):
        """测试创建推送器"""
        streamer = create_pnl_streamer(
            update_interval=0.5,
        )

        assert streamer.config.update_interval_seconds == 0.5

    def test_create_pnl_streamer_with_services(self):
        """测试带服务创建推送器"""
        mock_portfolio_service = MagicMock()
        mock_price_service = MagicMock()

        streamer = create_pnl_streamer(
            portfolio_service=mock_portfolio_service,
            price_service=mock_price_service,
        )

        assert streamer.portfolio_service == mock_portfolio_service
        assert streamer.price_service == mock_price_service


# ============ 边缘情况测试 ============

class TestEdgeCases:
    """测试边缘情况"""

    @pytest.mark.asyncio
    async def test_subscribe_before_start(self, streamer, mock_websocket):
        """测试启动前订阅"""
        # 不启动就订阅，应该能成功
        success = await streamer.subscribe_portfolio(mock_websocket, 1)

        assert success

    @pytest.mark.asyncio
    async def test_empty_portfolio_id_list(self, streamer):
        """测试空订阅列表"""
        await streamer.start()

        # 没有订阅者时更新循环不应崩溃
        await asyncio.sleep(0.3)

        assert streamer.state == StreamerState.RUNNING

        await streamer.stop()

    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, streamer):
        """测试并发订阅"""
        await streamer.start()

        # 并发订阅多个
        tasks = []
        for i in range(10):
            ws = MockWebSocket()
            tasks.append(streamer.subscribe_portfolio(ws, i))

        results = await asyncio.gather(*tasks)

        assert all(results)
        assert streamer.subscriber_count == 10

        await streamer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
