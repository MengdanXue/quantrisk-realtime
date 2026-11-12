"""
实时P&L推送引擎 - PnL Streamer

提供投资组合和持仓的实时盈亏推送能力。

核心功能:
1. 实时计算持仓P&L变化
2. 通过WebSocket推送P&L更新
3. 支持投资组合级别和持仓级别订阅
4. 增量更新优化(只推送变化的数据)
5. 与MarginMonitor和AlertService集成

使用方式:
    streamer = PnLStreamer(portfolio_service, connection_manager)
    await streamer.start()

    # 客户端订阅
    await streamer.subscribe_portfolio(websocket, portfolio_id)

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import json
from collections import defaultdict

from quantrisk_realtime.core.mock_guard import MockDataDisabledError, require_real_data

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class SubscriptionType(str, Enum):
    """订阅类型"""
    PORTFOLIO = "portfolio"     # 投资组合级别
    POSITION = "position"       # 单持仓级别
    ACCOUNT = "account"         # 账户级别(所有投资组合)


class UpdateType(str, Enum):
    """更新类型"""
    FULL = "full"               # 全量更新
    INCREMENTAL = "incremental" # 增量更新
    HEARTBEAT = "heartbeat"     # 心跳


class StreamerState(str, Enum):
    """推送器状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ 数据类定义 ============

@dataclass
class PositionPnL:
    """单持仓P&L数据"""
    position_id: int
    symbol: str

    # 价格
    current_price: float = 0.0
    average_cost: float = 0.0

    # 数量
    quantity: float = 0.0
    direction: int = 1  # 1=多, -1=空

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl_today: float = 0.0

    # 市值
    market_value: float = 0.0
    cost_basis: float = 0.0

    # 时间戳
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'average_cost': self.average_cost,
            'quantity': self.quantity,
            'direction': self.direction,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl_today': self.realized_pnl_today,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class PortfolioPnL:
    """投资组合P&L数据"""
    portfolio_id: int
    account_id: str = ""

    # 汇总P&L
    total_unrealized_pnl: float = 0.0
    total_realized_pnl_today: float = 0.0
    total_pnl_today: float = 0.0

    # 汇总市值
    total_market_value: float = 0.0
    total_cost_basis: float = 0.0
    equity: float = 0.0
    cash: float = 0.0

    # 回报率
    daily_return_pct: float = 0.0
    total_return_pct: float = 0.0

    # 风险指标
    margin_ratio: float = 0.0
    var_1d: float = 0.0

    # 持仓数
    position_count: int = 0

    # 持仓详情(可选)
    positions: List[PositionPnL] = field(default_factory=list)

    # 时间戳
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self, include_positions: bool = False) -> Dict[str, Any]:
        result = {
            'portfolio_id': self.portfolio_id,
            'account_id': self.account_id,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl_today': self.total_realized_pnl_today,
            'total_pnl_today': self.total_pnl_today,
            'total_market_value': self.total_market_value,
            'total_cost_basis': self.total_cost_basis,
            'equity': self.equity,
            'cash': self.cash,
            'daily_return_pct': self.daily_return_pct,
            'total_return_pct': self.total_return_pct,
            'margin_ratio': self.margin_ratio,
            'var_1d': self.var_1d,
            'position_count': self.position_count,
            'updated_at': self.updated_at.isoformat(),
        }
        if include_positions:
            result['positions'] = [p.to_dict() for p in self.positions]
        return result


@dataclass
class PnLUpdate:
    """P&L更新消息"""
    update_type: UpdateType
    subscription_type: SubscriptionType
    timestamp: datetime = field(default_factory=datetime.now)

    # 数据
    portfolio_pnl: Optional[PortfolioPnL] = None
    position_pnl: Optional[PositionPnL] = None

    # 变化量(增量更新时使用)
    pnl_change: float = 0.0
    price_change: float = 0.0

    # 序列号(用于客户端检测丢包)
    sequence: int = 0

    def to_json(self) -> str:
        data = {
            'type': 'pnl_update',
            'update_type': self.update_type.value,
            'subscription_type': self.subscription_type.value,
            'timestamp': self.timestamp.isoformat(),
            'sequence': self.sequence,
        }

        if self.portfolio_pnl:
            data['portfolio'] = self.portfolio_pnl.to_dict(include_positions=True)

        if self.position_pnl:
            data['position'] = self.position_pnl.to_dict()

        if self.update_type == UpdateType.INCREMENTAL:
            data['pnl_change'] = self.pnl_change
            data['price_change'] = self.price_change

        return json.dumps(data)


@dataclass
class StreamerConfig:
    """推送器配置"""
    # 更新间隔
    update_interval_seconds: float = 1.0      # 默认每秒更新
    heartbeat_interval_seconds: float = 30.0  # 心跳间隔

    # 增量更新阈值
    min_pnl_change_threshold: float = 0.01    # P&L变化超过1%才推送
    min_price_change_threshold: float = 0.001 # 价格变化超过0.1%才推送

    # 批量配置
    max_updates_per_batch: int = 100          # 每批最多更新数
    batch_delay_ms: int = 50                  # 批量延迟(毫秒)

    # 连接配置
    max_subscriptions_per_client: int = 50    # 每客户端最大订阅数

    # 缓存配置
    cache_ttl_seconds: int = 5                # P&L缓存TTL


# ============ 主类: P&L推送引擎 ============

class PnLStreamer:
    """
    实时P&L推送引擎

    核心职责:
    1. 监听价格变化
    2. 计算实时P&L
    3. 推送更新到订阅客户端
    4. 管理订阅关系

    使用示例:
        streamer = PnLStreamer(
            portfolio_service=portfolio_service,
            price_service=price_service,
        )

        await streamer.start()

        # 客户端订阅
        await streamer.subscribe_portfolio(websocket, portfolio_id)

        # 停止
        await streamer.stop()
    """

    def __init__(
        self,
        portfolio_service: Any = None,
        price_service: Any = None,
        config: Optional[StreamerConfig] = None,
        on_pnl_update: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """
        初始化推送引擎

        Args:
            portfolio_service: 投资组合服务(获取持仓数据)
            price_service: 价格服务(获取实时价格)
            config: 推送器配置
            on_pnl_update: P&L更新回调
            on_error: 错误回调
        """
        self.portfolio_service = portfolio_service
        self.price_service = price_service
        self.config = config or StreamerConfig()
        self.on_pnl_update = on_pnl_update
        self.on_error = on_error

        # 状态
        self._state = StreamerState.STOPPED
        self._update_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # 订阅管理
        # portfolio_id -> Set[websocket]
        self._portfolio_subscribers: Dict[int, Set[Any]] = defaultdict(set)
        # position_id -> Set[websocket]
        self._position_subscribers: Dict[int, Set[Any]] = defaultdict(set)
        # account_id -> Set[websocket]
        self._account_subscribers: Dict[str, Set[Any]] = defaultdict(set)
        # websocket -> Set[subscription_keys]
        self._client_subscriptions: Dict[Any, Set[str]] = defaultdict(set)

        # P&L缓存
        self._portfolio_pnl_cache: Dict[int, PortfolioPnL] = {}
        self._position_pnl_cache: Dict[int, PositionPnL] = {}

        # 统计
        self._sequence = 0
        self._update_count = 0
        self._error_count = 0
        self._start_time: Optional[datetime] = None

    @property
    def state(self) -> StreamerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == StreamerState.RUNNING

    @property
    def subscriber_count(self) -> int:
        """总订阅者数"""
        all_subs = set()
        for subs in self._portfolio_subscribers.values():
            all_subs.update(subs)
        for subs in self._position_subscribers.values():
            all_subs.update(subs)
        for subs in self._account_subscribers.values():
            all_subs.update(subs)
        return len(all_subs)

    # ============ 生命周期管理 ============

    async def start(self):
        """启动推送引擎"""
        if self._state == StreamerState.RUNNING:
            logger.warning("PnL streamer is already running")
            return

        logger.info("Starting PnL streamer...")
        self._state = StreamerState.STARTING
        self._start_time = datetime.now()

        try:
            # 启动更新循环
            self._update_task = asyncio.create_task(self._update_loop())
            # 启动心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self._state = StreamerState.RUNNING
            logger.info("PnL streamer started successfully")

        except Exception as e:
            self._state = StreamerState.ERROR
            logger.error(f"Failed to start PnL streamer: {e}")
            raise

    async def stop(self):
        """停止推送引擎"""
        if self._state == StreamerState.STOPPED:
            return

        logger.info("Stopping PnL streamer...")
        self._state = StreamerState.STOPPED

        # 取消任务
        for task in [self._update_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._update_task = None
        self._heartbeat_task = None

        # 清理订阅
        self._portfolio_subscribers.clear()
        self._position_subscribers.clear()
        self._account_subscribers.clear()
        self._client_subscriptions.clear()

        logger.info("PnL streamer stopped")

    async def pause(self):
        """暂停推送"""
        if self._state == StreamerState.RUNNING:
            self._state = StreamerState.PAUSED
            logger.info("PnL streamer paused")

    async def resume(self):
        """恢复推送"""
        if self._state == StreamerState.PAUSED:
            self._state = StreamerState.RUNNING
            logger.info("PnL streamer resumed")

    # ============ 订阅管理 ============

    async def subscribe_portfolio(
        self,
        websocket: Any,
        portfolio_id: int
    ) -> bool:
        """
        订阅投资组合P&L更新

        Args:
            websocket: WebSocket连接
            portfolio_id: 投资组合ID

        Returns:
            是否订阅成功
        """
        # 检查订阅限制
        if len(self._client_subscriptions[websocket]) >= self.config.max_subscriptions_per_client:
            logger.warning(f"Client exceeded max subscriptions limit")
            return False

        sub_key = f"portfolio:{portfolio_id}"

        self._portfolio_subscribers[portfolio_id].add(websocket)
        self._client_subscriptions[websocket].add(sub_key)

        logger.info(f"Client subscribed to portfolio {portfolio_id}")

        # 发送初始全量数据
        await self._send_initial_portfolio_data(websocket, portfolio_id)

        return True

    async def subscribe_position(
        self,
        websocket: Any,
        position_id: int
    ) -> bool:
        """订阅单持仓P&L更新"""
        if len(self._client_subscriptions[websocket]) >= self.config.max_subscriptions_per_client:
            return False

        sub_key = f"position:{position_id}"

        self._position_subscribers[position_id].add(websocket)
        self._client_subscriptions[websocket].add(sub_key)

        logger.info(f"Client subscribed to position {position_id}")
        return True

    async def subscribe_account(
        self,
        websocket: Any,
        account_id: str
    ) -> bool:
        """订阅账户级别P&L更新(所有投资组合)"""
        if len(self._client_subscriptions[websocket]) >= self.config.max_subscriptions_per_client:
            return False

        sub_key = f"account:{account_id}"

        self._account_subscribers[account_id].add(websocket)
        self._client_subscriptions[websocket].add(sub_key)

        logger.info(f"Client subscribed to account {account_id}")
        return True

    async def unsubscribe(
        self,
        websocket: Any,
        subscription_key: Optional[str] = None
    ):
        """
        取消订阅

        Args:
            websocket: WebSocket连接
            subscription_key: 特定订阅键(None=取消所有)
        """
        if subscription_key:
            # 取消特定订阅
            self._remove_subscription(websocket, subscription_key)
        else:
            # 取消所有订阅
            keys = list(self._client_subscriptions[websocket])
            for key in keys:
                self._remove_subscription(websocket, key)

    def _remove_subscription(self, websocket: Any, sub_key: str):
        """移除单个订阅"""
        parts = sub_key.split(':')
        if len(parts) != 2:
            return

        sub_type, sub_id = parts[0], parts[1]

        if sub_type == 'portfolio':
            portfolio_id = int(sub_id)
            self._portfolio_subscribers[portfolio_id].discard(websocket)
        elif sub_type == 'position':
            position_id = int(sub_id)
            self._position_subscribers[position_id].discard(websocket)
        elif sub_type == 'account':
            self._account_subscribers[sub_id].discard(websocket)

        self._client_subscriptions[websocket].discard(sub_key)

    async def handle_disconnect(self, websocket: Any):
        """处理客户端断开连接"""
        await self.unsubscribe(websocket)
        if websocket in self._client_subscriptions:
            del self._client_subscriptions[websocket]
        logger.info("Client disconnected, subscriptions cleaned up")

    # ============ 更新循环 ============

    async def _update_loop(self):
        """主更新循环"""
        logger.info("Entering PnL update loop")

        while self._state in [StreamerState.RUNNING, StreamerState.PAUSED]:
            try:
                if self._state == StreamerState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # 获取需要更新的投资组合
                portfolio_ids = list(self._portfolio_subscribers.keys())

                for portfolio_id in portfolio_ids:
                    if not self._portfolio_subscribers[portfolio_id]:
                        continue

                    # 计算P&L
                    pnl = await self._calculate_portfolio_pnl(portfolio_id)

                    if pnl:
                        # 检查是否有变化
                        if self._should_send_update(portfolio_id, pnl):
                            # 发送更新
                            await self._broadcast_portfolio_update(portfolio_id, pnl)
                            # 更新缓存
                            self._portfolio_pnl_cache[portfolio_id] = pnl

                # 等待下一次更新
                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in PnL update loop: {e}", exc_info=True)
                if self.on_error:
                    try:
                        await self.on_error(e)
                    except Exception as callback_err:
                        logger.warning(f"PnL error callback failed: {callback_err}")
                await asyncio.sleep(5)  # 错误后等待

        logger.info("Exiting PnL update loop")

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._state in [StreamerState.RUNNING, StreamerState.PAUSED]:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                if self._state != StreamerState.RUNNING:
                    continue

                # 发送心跳到所有客户端
                await self._broadcast_heartbeat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    # ============ P&L计算 ============

    async def _calculate_portfolio_pnl(self, portfolio_id: int) -> Optional[PortfolioPnL]:
        """计算投资组合P&L

        没有 portfolio_service 时不再改用随机数。缺少服务是一种配置状态，不是
        "要求模拟"的意思；把它当成模拟开关，会让一次接线失误变成向客户端推送
        编造的盈亏，而推送报文里没有任何地方能看出来。
        """
        try:
            if self.portfolio_service:
                # 从服务获取真实数据
                data = await self._fetch_portfolio_data(portfolio_id)
                return self._parse_portfolio_data(portfolio_id, data)

            require_real_data(
                f"P&L for portfolio {portfolio_id}",
                "Construct the streamer with a portfolio_service, e.g. "
                "create_pnl_streamer(portfolio_service=...).",
            )
            return await self._generate_mock_pnl(portfolio_id)

        except MockDataDisabledError:
            raise
        except Exception as e:
            logger.error(f"Failed to calculate P&L for portfolio {portfolio_id}: {e}")
            return None

    async def _fetch_portfolio_data(self, portfolio_id: int) -> Dict[str, Any]:
        """从服务获取投资组合数据

        服务缺少 get_portfolio_summary 时报错而不是返回 ``{}``：空字典经
        ``_parse_portfolio_data`` 会变成一份全 0 的 P&L，而全 0 在界面上读起来
        是"无敞口、无盈亏"，与真实结果无从分辨。
        """
        if not hasattr(self.portfolio_service, 'get_portfolio_summary'):
            raise AttributeError(
                f"{type(self.portfolio_service).__name__} has no get_portfolio_summary(); "
                "cannot produce P&L for portfolio "
                f"{portfolio_id} without inventing one."
            )
        return await self.portfolio_service.get_portfolio_summary(portfolio_id)

    def _parse_portfolio_data(
        self,
        portfolio_id: int,
        data: Dict[str, Any]
    ) -> PortfolioPnL:
        """解析投资组合数据"""
        pnl = PortfolioPnL(portfolio_id=portfolio_id)

        pnl.total_unrealized_pnl = float(data.get('unrealized_pnl', 0))
        pnl.total_realized_pnl_today = float(data.get('realized_pnl_today', 0))
        pnl.total_pnl_today = pnl.total_unrealized_pnl + pnl.total_realized_pnl_today

        pnl.total_market_value = float(data.get('total_market_value', 0))
        pnl.total_cost_basis = float(data.get('total_cost_basis', 0))
        pnl.equity = float(data.get('equity', 0))
        pnl.cash = float(data.get('cash', 0))

        pnl.position_count = int(data.get('position_count', 0))
        pnl.margin_ratio = float(data.get('margin_ratio', 0))

        # 计算回报率
        if pnl.total_cost_basis > 0:
            pnl.total_return_pct = (pnl.total_unrealized_pnl / pnl.total_cost_basis) * 100

        # 解析持仓
        positions_data = data.get('positions', [])
        for pos_data in positions_data:
            pos_pnl = PositionPnL(
                position_id=pos_data.get('position_id', 0),
                symbol=pos_data.get('symbol', ''),
                current_price=float(pos_data.get('current_price', 0)),
                average_cost=float(pos_data.get('average_cost', 0)),
                quantity=float(pos_data.get('quantity', 0)),
                direction=int(pos_data.get('direction', 1)),
                unrealized_pnl=float(pos_data.get('unrealized_pnl', 0)),
                market_value=float(pos_data.get('market_value', 0)),
            )
            pnl.positions.append(pos_pnl)

        pnl.updated_at = datetime.now()
        return pnl

    async def _generate_mock_pnl(self, portfolio_id: int) -> PortfolioPnL:
        """生成模拟P&L数据(测试用)"""
        import random

        # 获取或创建缓存
        cached = self._portfolio_pnl_cache.get(portfolio_id)

        if cached:
            # 基于缓存生成变化
            pnl = PortfolioPnL(portfolio_id=portfolio_id)
            pnl.total_unrealized_pnl = cached.total_unrealized_pnl + random.uniform(-100, 100)
            pnl.total_realized_pnl_today = cached.total_realized_pnl_today
            pnl.total_market_value = cached.total_market_value + random.uniform(-500, 500)
            pnl.total_cost_basis = cached.total_cost_basis
            pnl.equity = cached.equity + random.uniform(-200, 200)
            pnl.cash = cached.cash
            pnl.position_count = cached.position_count
        else:
            # 初始化
            pnl = PortfolioPnL(portfolio_id=portfolio_id)
            pnl.total_unrealized_pnl = random.uniform(-5000, 10000)
            pnl.total_realized_pnl_today = random.uniform(-1000, 2000)
            pnl.total_market_value = random.uniform(50000, 200000)
            pnl.total_cost_basis = pnl.total_market_value - pnl.total_unrealized_pnl
            pnl.equity = pnl.total_market_value + random.uniform(10000, 50000)
            pnl.cash = pnl.equity - pnl.total_market_value
            pnl.position_count = random.randint(5, 20)

            # 添加模拟持仓
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            for i, symbol in enumerate(symbols[:pnl.position_count]):
                pos = PositionPnL(
                    position_id=portfolio_id * 100 + i,
                    symbol=symbol,
                    current_price=random.uniform(100, 500),
                    quantity=random.randint(10, 100),
                    unrealized_pnl=random.uniform(-500, 1000),
                )
                pos.average_cost = pos.current_price - (pos.unrealized_pnl / pos.quantity)
                pos.market_value = pos.current_price * pos.quantity
                pnl.positions.append(pos)

        pnl.total_pnl_today = pnl.total_unrealized_pnl + pnl.total_realized_pnl_today

        if pnl.total_cost_basis > 0:
            pnl.total_return_pct = (pnl.total_unrealized_pnl / pnl.total_cost_basis) * 100

        pnl.updated_at = datetime.now()
        return pnl

    # ============ 推送逻辑 ============

    def _should_send_update(
        self,
        portfolio_id: int,
        new_pnl: PortfolioPnL
    ) -> bool:
        """判断是否应该发送更新"""
        cached = self._portfolio_pnl_cache.get(portfolio_id)

        if not cached:
            return True  # 没有缓存,发送全量

        # 检查P&L变化
        pnl_change = abs(new_pnl.total_unrealized_pnl - cached.total_unrealized_pnl)
        if cached.total_unrealized_pnl != 0:
            pnl_change_pct = pnl_change / abs(cached.total_unrealized_pnl)
            if pnl_change_pct >= self.config.min_pnl_change_threshold:
                return True
        elif pnl_change > 0:
            return True

        # 检查市值变化
        value_change = abs(new_pnl.total_market_value - cached.total_market_value)
        if cached.total_market_value > 0:
            value_change_pct = value_change / cached.total_market_value
            if value_change_pct >= self.config.min_price_change_threshold:
                return True

        return False

    async def _broadcast_portfolio_update(
        self,
        portfolio_id: int,
        pnl: PortfolioPnL
    ):
        """广播投资组合更新"""
        subscribers = self._portfolio_subscribers.get(portfolio_id, set())

        if not subscribers:
            return

        # 构建更新消息
        self._sequence += 1
        update = PnLUpdate(
            update_type=UpdateType.INCREMENTAL,
            subscription_type=SubscriptionType.PORTFOLIO,
            portfolio_pnl=pnl,
            sequence=self._sequence,
        )

        # 计算变化量
        cached = self._portfolio_pnl_cache.get(portfolio_id)
        if cached:
            update.pnl_change = pnl.total_unrealized_pnl - cached.total_unrealized_pnl

        message = update.to_json()

        # 发送给所有订阅者
        dead_connections = []
        for ws in subscribers:
            try:
                await self._send_message(ws, message)
            except Exception as e:
                logger.warning(f"Failed to send to subscriber: {e}")
                dead_connections.append(ws)

        # 清理断开的连接
        for ws in dead_connections:
            await self.handle_disconnect(ws)

        self._update_count += 1

        # 触发回调
        if self.on_pnl_update:
            try:
                await self.on_pnl_update(update)
            except Exception as e:
                logger.error(f"PnL update callback failed: {e}")

    async def _send_initial_portfolio_data(
        self,
        websocket: Any,
        portfolio_id: int
    ):
        """发送初始全量数据"""
        pnl = await self._calculate_portfolio_pnl(portfolio_id)

        if pnl:
            self._sequence += 1
            update = PnLUpdate(
                update_type=UpdateType.FULL,
                subscription_type=SubscriptionType.PORTFOLIO,
                portfolio_pnl=pnl,
                sequence=self._sequence,
            )

            try:
                await self._send_message(websocket, update.to_json())
            except Exception as e:
                logger.error(f"Failed to send initial data: {e}")

    async def _broadcast_heartbeat(self):
        """广播心跳"""
        self._sequence += 1

        heartbeat = {
            'type': 'pnl_heartbeat',
            'timestamp': datetime.now().isoformat(),
            'sequence': self._sequence,
            'subscriber_count': self.subscriber_count,
        }

        message = json.dumps(heartbeat)

        # 收集所有订阅者
        all_subscribers = set()
        for subs in self._portfolio_subscribers.values():
            all_subscribers.update(subs)
        for subs in self._position_subscribers.values():
            all_subscribers.update(subs)
        for subs in self._account_subscribers.values():
            all_subscribers.update(subs)

        dead_connections = []
        for ws in all_subscribers:
            try:
                await self._send_message(ws, message)
            except Exception as e:
                logger.debug(f"Failed to send heartbeat to subscriber: {e}")
                dead_connections.append(ws)

        for ws in dead_connections:
            await self.handle_disconnect(ws)

    async def _send_message(self, websocket: Any, message: str):
        """发送消息到WebSocket"""
        if hasattr(websocket, 'send_text'):
            await websocket.send_text(message)
        elif hasattr(websocket, 'send'):
            await websocket.send(message)
        else:
            raise ValueError("Invalid websocket object")

    # ============ 统计信息 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            'state': self._state.value,
            'uptime_seconds': uptime,
            'subscriber_count': self.subscriber_count,
            'portfolio_subscriptions': len(self._portfolio_subscribers),
            'position_subscriptions': len(self._position_subscribers),
            'account_subscriptions': len(self._account_subscribers),
            'update_count': self._update_count,
            'error_count': self._error_count,
            'sequence': self._sequence,
            'config': {
                'update_interval': self.config.update_interval_seconds,
                'heartbeat_interval': self.config.heartbeat_interval_seconds,
            }
        }


# ============ 工厂函数 ============

def create_pnl_streamer(
    portfolio_service: Any = None,
    price_service: Any = None,
    update_interval: float = 1.0,
) -> PnLStreamer:
    """
    创建P&L推送引擎

    Args:
        portfolio_service: 投资组合服务
        price_service: 价格服务
        update_interval: 更新间隔(秒)

    Returns:
        配置好的PnLStreamer实例
    """
    config = StreamerConfig(
        update_interval_seconds=update_interval,
    )

    return PnLStreamer(
        portfolio_service=portfolio_service,
        price_service=price_service,
        config=config,
    )
