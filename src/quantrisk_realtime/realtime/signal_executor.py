"""
策略信号执行器
Strategy Signal Executor

将策略信号转化为实际订单执行:
- 接收融合信号
- 计算目标仓位
- 生成交易订单
- 调用Java后端执行
- 跟踪执行状态

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class SignalDirection(str, Enum):
    """信号方向"""
    STRONG_LONG = "strong_long"    # 强烈看多
    LONG = "long"                  # 看多
    NEUTRAL = "neutral"            # 中性
    SHORT = "short"                # 看空
    STRONG_SHORT = "strong_short"  # 强烈看空


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"          # 市价单
    LIMIT = "limit"            # 限价单
    STOP = "stop"              # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"         # 待执行
    SUBMITTED = "submitted"     # 已提交
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 全部成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 被拒绝
    FAILED = "failed"           # 执行失败


class ExecutionMode(str, Enum):
    """执行模式"""
    PAPER = "paper"        # 模拟交易
    LIVE = "live"          # 实盘交易
    SIGNAL_ONLY = "signal" # 仅生成信号


# ============ 数据类 ============

@dataclass
class TradingSignal:
    """交易信号"""
    signal_id: str
    symbol: str
    direction: SignalDirection
    strength: float                # 信号强度 0-1
    confidence: float              # 置信度 0-1
    source: str                    # 信号来源
    generated_at: datetime

    # 可选字段
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: str = "intraday"  # intraday, swing, position
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_value(self) -> float:
        """标准化信号值 [-1, 1]"""
        direction_map = {
            SignalDirection.STRONG_LONG: 1.0,
            SignalDirection.LONG: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SHORT: -0.5,
            SignalDirection.STRONG_SHORT: -1.0,
        }
        base = direction_map.get(self.direction, 0.0)
        return base * self.strength * self.confidence


@dataclass
class TradeOrder:
    """交易订单"""
    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # 执行参数
    portfolio_id: Optional[int] = None
    currency: str = "USD"
    multiplier: float = 1.0
    margin_rate: Optional[float] = None

    # 状态
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0

    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # 消息
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'portfolio_id': self.portfolio_id,
            'currency': self.currency,
            'multiplier': self.multiplier,
            'margin_rate': self.margin_rate,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'message': self.message,
        }


@dataclass
class PositionTarget:
    """目标仓位"""
    symbol: str
    target_weight: float           # 目标权重 [-1, 1]
    current_weight: float          # 当前权重
    target_quantity: float         # 目标数量
    current_quantity: float        # 当前数量
    quantity_delta: float          # 变化数量
    action: str                    # buy/sell/hold
    reason: str                    # 调仓原因


@dataclass
class ExecutorConfig:
    """执行器配置"""
    # API配置
    java_api_base_url: str = "http://localhost:8080"
    auth_token: str = ""

    # 执行模式
    execution_mode: ExecutionMode = ExecutionMode.PAPER

    # 仓位限制
    max_position_weight: float = 0.20      # 单仓位最大权重
    max_total_exposure: float = 1.0        # 最大总敞口
    min_trade_value: float = 100.0         # 最小交易金额

    # 信号阈值
    min_signal_strength: float = 0.3       # 最小信号强度
    min_signal_confidence: float = 0.5     # 最小信号置信度
    signal_decay_hours: float = 24.0       # 信号有效期(小时)

    # 风险控制
    max_daily_trades: int = 50             # 日最大交易次数
    max_daily_turnover: float = 2.0        # 日最大换手率
    position_sizing_method: str = "equal"  # equal, volatility, kelly

    # 执行控制
    order_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # 日志
    log_orders: bool = True
    log_signals: bool = True


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    order: TradeOrder
    executed_at: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'order': self.order.to_dict(),
            'executed_at': self.executed_at.isoformat(),
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'slippage': self.slippage,
            'message': self.message,
        }


# ============ 仓位管理器 ============

class PositionSizer(ABC):
    """仓位计算器抽象基类"""

    @abstractmethod
    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_positions: Dict[str, float],
        config: ExecutorConfig,
    ) -> float:
        """计算仓位大小"""
        pass


class EqualWeightSizer(PositionSizer):
    """等权重仓位计算器"""

    def __init__(self, base_weight: float = 0.05):
        self.base_weight = base_weight

    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_positions: Dict[str, float],
        config: ExecutorConfig,
    ) -> float:
        # 基于信号强度调整权重
        weight = self.base_weight * signal.strength * signal.confidence

        # 应用最大权重限制
        weight = min(weight, config.max_position_weight)

        # 计算目标金额
        target_value = portfolio_value * weight

        return target_value


class VolatilityAdjustedSizer(PositionSizer):
    """波动率调整仓位计算器"""

    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback
        self._volatilities: Dict[str, float] = {}

    def set_volatility(self, symbol: str, vol: float):
        """设置标的波动率"""
        self._volatilities[symbol] = vol

    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_positions: Dict[str, float],
        config: ExecutorConfig,
    ) -> float:
        # 获取波动率
        vol = self._volatilities.get(signal.symbol, 0.20)  # 默认20%

        # 计算波动率调整系数
        vol_scalar = self.target_vol / max(vol, 0.01)
        vol_scalar = min(vol_scalar, 3.0)  # 最多3倍

        # 基于信号和波动率计算权重
        base_weight = 0.05 * signal.strength * signal.confidence
        adjusted_weight = base_weight * vol_scalar

        # 应用限制
        adjusted_weight = min(adjusted_weight, config.max_position_weight)

        return portfolio_value * adjusted_weight


# ============ 信号执行器 ============

class SignalExecutor:
    """策略信号执行器"""

    def __init__(
        self,
        config: ExecutorConfig,
        position_sizer: Optional[PositionSizer] = None,
        on_order_submitted: Optional[Callable[[TradeOrder], Awaitable[None]]] = None,
        on_order_filled: Optional[Callable[[ExecutionResult], Awaitable[None]]] = None,
        on_order_rejected: Optional[Callable[[TradeOrder, str], Awaitable[None]]] = None,
    ):
        self.config = config
        self.position_sizer = position_sizer or EqualWeightSizer()

        # 回调
        self.on_order_submitted = on_order_submitted
        self.on_order_filled = on_order_filled
        self.on_order_rejected = on_order_rejected

        # HTTP客户端
        self._session: Optional[aiohttp.ClientSession] = None

        # 状态
        self._is_running = False
        self._pending_orders: Dict[str, TradeOrder] = {}
        self._order_history: List[TradeOrder] = []

        # 统计
        self._daily_trade_count = 0
        self._daily_turnover = 0.0
        self._last_reset_date = datetime.now().date()

        # 信号队列
        self._signal_queue: asyncio.Queue = asyncio.Queue()
        self._process_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self):
        """启动执行器"""
        if self._is_running:
            return

        self._is_running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.order_timeout_seconds)
        )

        # 启动信号处理任务
        self._process_task = asyncio.create_task(self._process_signals())

        logger.info(f"SignalExecutor started in {self.config.execution_mode.value} mode")

    async def stop(self):
        """停止执行器"""
        if not self._is_running:
            return

        self._is_running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("SignalExecutor stopped")

    async def submit_signal(self, signal: TradingSignal) -> bool:
        """
        提交交易信号

        Returns:
            是否接受信号
        """
        # 验证信号
        if not self._validate_signal(signal):
            logger.warning(f"Signal rejected: {signal.signal_id} - validation failed")
            return False

        # 加入队列
        await self._signal_queue.put(signal)

        if self.config.log_signals:
            logger.info(f"Signal queued: {signal.symbol} {signal.direction.value} "
                       f"strength={signal.strength:.2f} confidence={signal.confidence:.2f}")

        return True

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """验证信号有效性"""
        # 检查信号强度
        if signal.strength < self.config.min_signal_strength:
            return False

        # 检查置信度
        if signal.confidence < self.config.min_signal_confidence:
            return False

        # 检查信号时效
        age = datetime.now() - signal.generated_at
        if age.total_seconds() > self.config.signal_decay_hours * 3600:
            return False

        # 中性信号不生成订单
        if signal.direction == SignalDirection.NEUTRAL:
            return False

        return True

    async def _process_signals(self):
        """处理信号队列"""
        while self._is_running:
            try:
                # 等待信号
                signal = await asyncio.wait_for(
                    self._signal_queue.get(),
                    timeout=1.0
                )

                # 重置日统计
                self._check_daily_reset()

                # 检查日限制
                if self._daily_trade_count >= self.config.max_daily_trades:
                    logger.warning("Daily trade limit reached")
                    continue

                # 处理信号
                result = await self._execute_signal(signal)

                if result:
                    self._daily_trade_count += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing signal: {e}")

    def _check_daily_reset(self):
        """检查并重置日统计"""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._daily_trade_count = 0
            self._daily_turnover = 0.0
            self._last_reset_date = today

    async def _execute_signal(
        self,
        signal: TradingSignal,
        portfolio_id: Optional[int] = None,
        portfolio_value: float = 1000000.0,
        current_price: Optional[float] = None,
    ) -> Optional[ExecutionResult]:
        """
        执行单个信号

        Args:
            signal: 交易信号
            portfolio_id: 投资组合ID
            portfolio_value: 组合价值
            current_price: 当前价格

        Returns:
            执行结果
        """
        try:
            # 计算仓位大小
            target_value = self.position_sizer.calculate_position_size(
                signal=signal,
                portfolio_value=portfolio_value,
                current_positions={},
                config=self.config,
            )

            # 检查最小交易金额
            if target_value < self.config.min_trade_value:
                logger.debug(f"Trade value too small: {target_value:.2f}")
                return None

            # 确定订单方向
            if signal.direction in [SignalDirection.STRONG_LONG, SignalDirection.LONG]:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # 计算数量（如果有价格）
            if current_price and current_price > 0:
                quantity = target_value / current_price
            else:
                quantity = target_value  # 使用金额作为代理

            # 创建订单
            order = TradeOrder(
                order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                portfolio_id=portfolio_id,
                limit_price=signal.target_price,
                stop_price=signal.stop_loss,
            )

            # 根据执行模式处理
            if self.config.execution_mode == ExecutionMode.SIGNAL_ONLY:
                # 仅生成信号，不执行
                order.status = OrderStatus.PENDING
                order.message = "Signal only mode - not executed"
                self._order_history.append(order)
                return ExecutionResult(success=True, order=order, message="Signal recorded")

            elif self.config.execution_mode == ExecutionMode.PAPER:
                # 模拟执行
                result = await self._paper_execute(order, current_price)
                return result

            else:
                # 实盘执行
                result = await self._live_execute(order)
                return result

        except Exception as e:
            logger.error(f"Error executing signal {signal.signal_id}: {e}")
            return None

    async def _paper_execute(
        self,
        order: TradeOrder,
        price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        模拟执行

        Args:
            order: 订单
            price: 执行价格

        Returns:
            执行结果
        """
        # 模拟成交
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = price or 100.0  # 默认价格
        order.filled_at = datetime.now()
        order.submitted_at = datetime.now()
        order.commission = order.filled_quantity * order.filled_price * 0.001  # 0.1%手续费
        order.message = "Paper trade executed"

        self._order_history.append(order)
        self._pending_orders.pop(order.order_id, None)

        result = ExecutionResult(
            success=True,
            order=order,
            commission=order.commission,
            message="Paper trade completed",
        )

        if self.config.log_orders:
            logger.info(f"Paper trade: {order.side.value} {order.quantity:.2f} {order.symbol} "
                       f"@ {order.filled_price:.2f}")

        # 触发回调
        if self.on_order_filled:
            await self.on_order_filled(result)

        return result

    async def _live_execute(self, order: TradeOrder) -> ExecutionResult:
        """
        实盘执行（调用Java后端）

        Args:
            order: 订单

        Returns:
            执行结果
        """
        if not self._session:
            return ExecutionResult(
                success=False,
                order=order,
                message="HTTP session not initialized"
            )

        # 记录待处理订单
        self._pending_orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        # 触发提交回调
        if self.on_order_submitted:
            await self.on_order_submitted(order)

        try:
            # 根据订单方向调用不同API
            if order.side == OrderSide.BUY:
                result = await self._call_buy_api(order)
            else:
                result = await self._call_sell_api(order)

            return result

        except Exception as e:
            logger.warning(
                "Order %s (%s %s) failed at the broker API: %s",
                order.order_id, order.side, getattr(order, "symbol", "<unknown>"), e,
                exc_info=True
            )
            order.status = OrderStatus.FAILED
            order.message = str(e)
            self._order_history.append(order)
            self._pending_orders.pop(order.order_id, None)

            if self.on_order_rejected:
                await self.on_order_rejected(order, str(e))

            return ExecutionResult(
                success=False,
                order=order,
                message=str(e)
            )

    async def _call_buy_api(self, order: TradeOrder) -> ExecutionResult:
        """调用买入API"""
        url = f"{self.config.java_api_base_url}/positions/buy"

        payload = {
            'portfolioId': order.portfolio_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': order.limit_price or order.filled_price or 0,
            'currency': order.currency,
            'multiplier': order.multiplier,
            'direction': 1 if order.side == OrderSide.BUY else -1,
        }

        if order.margin_rate:
            payload['marginRate'] = order.margin_rate

        headers = {
            'Content-Type': 'application/json',
        }
        if self.config.auth_token:
            headers['Authorization'] = self.config.auth_token

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(url, json=payload, headers=headers) as resp:
                    data = await resp.json()

                    if resp.status == 200 and data.get('success'):
                        order.status = OrderStatus.FILLED
                        order.filled_at = datetime.now()
                        order.filled_quantity = order.quantity
                        order.message = data.get('message', 'Success')

                        self._order_history.append(order)
                        self._pending_orders.pop(order.order_id, None)

                        result = ExecutionResult(
                            success=True,
                            order=order,
                            message="Buy order executed",
                        )

                        if self.on_order_filled:
                            await self.on_order_filled(result)

                        return result
                    else:
                        error_msg = data.get('message', f'HTTP {resp.status}')
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                            continue

                        order.status = OrderStatus.REJECTED
                        order.message = error_msg
                        self._order_history.append(order)
                        self._pending_orders.pop(order.order_id, None)

                        if self.on_order_rejected:
                            await self.on_order_rejected(order, error_msg)

                        return ExecutionResult(
                            success=False,
                            order=order,
                            message=error_msg
                        )

            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                raise
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                raise

        # 不应该到达这里
        order.status = OrderStatus.FAILED
        return ExecutionResult(success=False, order=order, message="Max retries exceeded")

    async def _call_sell_api(self, order: TradeOrder) -> ExecutionResult:
        """调用卖出API"""
        url = f"{self.config.java_api_base_url}/positions/sell"

        payload = {
            'portfolioId': order.portfolio_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': order.limit_price or order.filled_price or 0,
        }

        headers = {
            'Content-Type': 'application/json',
        }
        if self.config.auth_token:
            headers['Authorization'] = self.config.auth_token

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(url, json=payload, headers=headers) as resp:
                    data = await resp.json()

                    if resp.status == 200 and data.get('success'):
                        order.status = OrderStatus.FILLED
                        order.filled_at = datetime.now()
                        order.filled_quantity = order.quantity
                        order.message = data.get('message', 'Success')

                        self._order_history.append(order)
                        self._pending_orders.pop(order.order_id, None)

                        result = ExecutionResult(
                            success=True,
                            order=order,
                            message="Sell order executed",
                        )

                        if self.on_order_filled:
                            await self.on_order_filled(result)

                        return result
                    else:
                        error_msg = data.get('message', f'HTTP {resp.status}')
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                            continue

                        order.status = OrderStatus.REJECTED
                        order.message = error_msg
                        self._order_history.append(order)
                        self._pending_orders.pop(order.order_id, None)

                        if self.on_order_rejected:
                            await self.on_order_rejected(order, error_msg)

                        return ExecutionResult(
                            success=False,
                            order=order,
                            message=error_msg
                        )

            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                raise
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                raise

        order.status = OrderStatus.FAILED
        return ExecutionResult(success=False, order=order, message="Max retries exceeded")

    # ============ 批量执行 ============

    async def execute_signals_batch(
        self,
        signals: List[TradingSignal],
        portfolio_id: int,
        portfolio_value: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[ExecutionResult]:
        """
        批量执行信号

        Args:
            signals: 信号列表
            portfolio_id: 投资组合ID
            portfolio_value: 组合价值
            prices: 价格字典 {symbol: price}

        Returns:
            执行结果列表
        """
        results = []
        prices = prices or {}

        for signal in signals:
            if not self._validate_signal(signal):
                continue

            price = prices.get(signal.symbol)
            result = await self._execute_signal(
                signal=signal,
                portfolio_id=portfolio_id,
                portfolio_value=portfolio_value,
                current_price=price,
            )

            if result:
                results.append(result)

        return results

    # ============ 查询方法 ============

    def get_pending_orders(self) -> List[TradeOrder]:
        """获取待处理订单"""
        return list(self._pending_orders.values())

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TradeOrder]:
        """获取订单历史"""
        orders = self._order_history

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if since:
            orders = [o for o in orders if o.created_at >= since]

        return orders[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        filled_orders = [o for o in self._order_history if o.status == OrderStatus.FILLED]
        rejected_orders = [o for o in self._order_history if o.status == OrderStatus.REJECTED]

        total_commission = sum(o.commission for o in filled_orders)

        return {
            'is_running': self._is_running,
            'execution_mode': self.config.execution_mode.value,
            'total_orders': len(self._order_history),
            'filled_orders': len(filled_orders),
            'rejected_orders': len(rejected_orders),
            'pending_orders': len(self._pending_orders),
            'total_commission': total_commission,
            'daily_trade_count': self._daily_trade_count,
            'daily_turnover': self._daily_turnover,
            'queue_size': self._signal_queue.qsize(),
        }


# ============ 工厂函数 ============

def create_signal_executor(
    java_api_url: str = "http://localhost:8080",
    auth_token: str = "",
    execution_mode: str = "paper",
    position_sizing: str = "equal",
    **kwargs,
) -> SignalExecutor:
    """
    创建信号执行器

    Args:
        java_api_url: Java后端API地址
        auth_token: 认证令牌
        execution_mode: 执行模式 (paper/live/signal)
        position_sizing: 仓位计算方法 (equal/volatility)
        **kwargs: 其他配置参数

    Returns:
        SignalExecutor实例
    """
    # 解析执行模式
    mode_map = {
        'paper': ExecutionMode.PAPER,
        'live': ExecutionMode.LIVE,
        'signal': ExecutionMode.SIGNAL_ONLY,
    }
    mode = mode_map.get(execution_mode.lower(), ExecutionMode.PAPER)

    # 创建配置
    config = ExecutorConfig(
        java_api_base_url=java_api_url,
        auth_token=auth_token,
        execution_mode=mode,
        **{k: v for k, v in kwargs.items() if hasattr(ExecutorConfig, k)},
    )

    # 创建仓位计算器
    if position_sizing == "volatility":
        sizer = VolatilityAdjustedSizer()
    else:
        sizer = EqualWeightSizer()

    return SignalExecutor(config=config, position_sizer=sizer)


def create_trading_signal(
    symbol: str,
    direction: str,
    strength: float,
    confidence: float,
    source: str = "strategy",
    **kwargs,
) -> TradingSignal:
    """
    创建交易信号

    Args:
        symbol: 标的代码
        direction: 方向 (strong_long/long/neutral/short/strong_short)
        strength: 强度 0-1
        confidence: 置信度 0-1
        source: 信号来源
        **kwargs: 其他参数

    Returns:
        TradingSignal实例
    """
    # 解析方向
    direction_map = {
        'strong_long': SignalDirection.STRONG_LONG,
        'long': SignalDirection.LONG,
        'neutral': SignalDirection.NEUTRAL,
        'short': SignalDirection.SHORT,
        'strong_short': SignalDirection.STRONG_SHORT,
    }
    signal_direction = direction_map.get(direction.lower(), SignalDirection.NEUTRAL)

    return TradingSignal(
        signal_id=f"SIG-{uuid.uuid4().hex[:8].upper()}",
        symbol=symbol,
        direction=signal_direction,
        strength=min(max(strength, 0), 1),  # 限制在0-1
        confidence=min(max(confidence, 0), 1),
        source=source,
        generated_at=datetime.now(),
        **{k: v for k, v in kwargs.items() if hasattr(TradingSignal, k)},
    )
