"""
策略回放器
Strategy Replayer

连接回测引擎与实时信号执行系统，实现：
- 历史信号回放
- 回测与实盘执行逻辑统一
- 策略绩效验证
- Walk-Forward验证支持

作者: TraderX Team
创建: 2025-12-24
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import asyncio

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    """回放模式"""
    BACKTEST = "backtest"       # 纯回测（不连接实时系统）
    PAPER = "paper"             # 模拟交易回放
    VALIDATION = "validation"   # 验证模式（对比回测与实际执行）
    WALK_FORWARD = "walk_forward"  # Walk-Forward优化


class ReplaySpeed(str, Enum):
    """回放速度"""
    INSTANT = "instant"         # 即时（尽快）
    REALTIME = "realtime"       # 1:1实时
    FAST = "fast"               # 10x加速
    SLOW = "slow"               # 0.5x减速


@dataclass
class ReplayConfig:
    """回放配置"""
    mode: ReplayMode = ReplayMode.BACKTEST
    speed: ReplaySpeed = ReplaySpeed.INSTANT

    # 执行配置
    initial_capital: float = 1_000_000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005

    # 信号配置
    min_signal_strength: float = 0.3
    min_signal_confidence: float = 0.5

    # 风控配置
    max_position_weight: float = 0.20
    max_daily_trades: int = 50
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    # 验证配置
    validation_tolerance: float = 0.01  # 1%容差
    save_detailed_log: bool = True


@dataclass
class ReplaySignal:
    """回放信号"""
    timestamp: datetime
    symbol: str
    direction: str  # strong_long/long/neutral/short/strong_short
    strength: float
    confidence: float
    source: str

    # 附加信息
    price_at_signal: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'source': self.source,
            'price_at_signal': self.price_at_signal,
            'metadata': self.metadata,
        }


@dataclass
class ReplayTrade:
    """回放交易"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    commission: float
    slippage: float
    signal: Optional[ReplaySignal] = None

    # 平仓时填充
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'signal': self.signal.to_dict() if self.signal else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
        }


@dataclass
class ReplayPosition:
    """回放持仓"""
    symbol: str
    side: str  # long/short/flat
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_price(self, price: float):
        """更新价格和未实现盈亏"""
        self.current_price = price
        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        elif self.side == 'short':
            self.unrealized_pnl = (self.entry_price - price) * self.quantity


@dataclass
class ReplayResult:
    """回放结果"""
    mode: ReplayMode
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # 资金曲线
    initial_capital: float
    final_capital: float
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission: float = 0.0

    # 绩效指标
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # 详细记录
    trades: List[ReplayTrade] = field(default_factory=list)
    signals_processed: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0

    # 验证结果 (validation模式)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'total_commission': self.total_commission,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'signals_processed': self.signals_processed,
            'signals_executed': self.signals_executed,
            'signals_rejected': self.signals_rejected,
            'validation_passed': self.validation_passed,
            'validation_errors': self.validation_errors,
        }


class StrategyReplayer:
    """
    策略回放器

    连接回测引擎与实时信号执行系统，实现回测与实盘逻辑统一。

    使用方式:
    ```python
    replayer = StrategyReplayer(config)

    # 加载历史信号
    replayer.load_signals(signals)

    # 加载价格数据
    replayer.load_prices(price_df)

    # 运行回放
    result = await replayer.run()
    ```
    """

    def __init__(self, config: Optional[ReplayConfig] = None):
        self.config = config or ReplayConfig()

        # 信号队列
        self._signals: List[ReplaySignal] = []
        self._signal_index = 0

        # 价格数据
        self._prices: Optional[pd.DataFrame] = None
        self._current_prices: Dict[str, float] = {}

        # 状态
        self._positions: Dict[str, ReplayPosition] = {}
        self._capital = self.config.initial_capital
        self._equity_curve: List[Dict[str, Any]] = []
        self._trades: List[ReplayTrade] = []
        self._trade_counter = 0
        self._daily_trades = 0
        self._current_date: Optional[datetime] = None

        # 统计
        self._signals_processed = 0
        self._signals_executed = 0
        self._signals_rejected = 0

        # 回放状态
        self._is_running = False
        self._current_timestamp: Optional[datetime] = None

        # 回调
        self._on_signal: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_position_update: Optional[Callable] = None

        # 信号执行器 (可选连接)
        self._signal_executor = None

    def load_signals(self, signals: List[Dict[str, Any]]) -> int:
        """
        加载历史信号

        Args:
            signals: 信号列表，每个信号包含:
                - timestamp: 时间戳
                - symbol: 标的代码
                - direction: 方向
                - strength: 强度
                - confidence: 置信度
                - source: 来源

        Returns:
            加载的信号数量
        """
        self._signals = []

        for sig in signals:
            timestamp = sig.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)

            replay_signal = ReplaySignal(
                timestamp=timestamp,
                symbol=sig['symbol'],
                direction=sig['direction'],
                strength=sig['strength'],
                confidence=sig['confidence'],
                source=sig.get('source', 'historical'),
                price_at_signal=sig.get('price'),
                metadata=sig.get('metadata', {}),
            )
            self._signals.append(replay_signal)

        # 按时间排序
        self._signals.sort(key=lambda x: x.timestamp)
        self._signal_index = 0

        logger.info(f"Loaded {len(self._signals)} signals")
        return len(self._signals)

    def load_prices(self, prices: pd.DataFrame) -> int:
        """
        加载价格数据

        Args:
            prices: 价格DataFrame，需要有datetime索引和symbol列
                    或者multi-index (datetime, symbol)

        Returns:
            数据点数量
        """
        if isinstance(prices.index, pd.MultiIndex):
            self._prices = prices
        else:
            # 假设是单个标的的价格序列
            if 'close' in prices.columns:
                self._prices = prices
            else:
                raise ValueError("Price DataFrame must have 'close' column")

        logger.info(f"Loaded {len(prices)} price records")
        return len(prices)

    def connect_executor(self, executor) -> None:
        """
        连接实时信号执行器

        在PAPER或VALIDATION模式下，可以连接真实的执行器进行比较
        """
        self._signal_executor = executor
        logger.info("Connected to signal executor")

    def set_callbacks(
        self,
        on_signal: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_position_update: Optional[Callable] = None,
    ):
        """设置回调函数"""
        self._on_signal = on_signal
        self._on_trade = on_trade
        self._on_position_update = on_position_update

    async def run(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ReplayResult:
        """
        运行回放

        Args:
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）

        Returns:
            ReplayResult 回放结果
        """
        run_start = datetime.now()
        self._is_running = True

        # 过滤信号时间范围
        signals_to_process = self._signals
        if start_time:
            signals_to_process = [s for s in signals_to_process if s.timestamp >= start_time]
        if end_time:
            signals_to_process = [s for s in signals_to_process if s.timestamp <= end_time]

        if not signals_to_process:
            logger.warning("No signals to process in the specified time range")
            return self._create_empty_result(run_start)

        # 重置状态
        self._reset_state()

        actual_start = signals_to_process[0].timestamp
        actual_end = signals_to_process[-1].timestamp

        logger.info(f"Starting replay from {actual_start} to {actual_end} with {len(signals_to_process)} signals")

        # 处理每个信号
        for signal in signals_to_process:
            if not self._is_running:
                break

            await self._process_signal(signal)

            # 速度控制
            if self.config.speed != ReplaySpeed.INSTANT:
                await self._apply_speed_delay(signal.timestamp)

        # 平掉所有持仓
        await self._close_all_positions()

        # 计算结果
        run_end = datetime.now()
        result = self._calculate_result(
            actual_start,
            actual_end,
            (run_end - run_start).total_seconds(),
        )

        self._is_running = False
        return result

    def stop(self):
        """停止回放"""
        self._is_running = False
        logger.info("Replay stopped")

    def _reset_state(self):
        """重置状态"""
        self._positions.clear()
        self._capital = self.config.initial_capital
        self._equity_curve = []
        self._trades = []
        self._trade_counter = 0
        self._daily_trades = 0
        self._current_date = None
        self._signals_processed = 0
        self._signals_executed = 0
        self._signals_rejected = 0
        self._current_prices.clear()

    async def _process_signal(self, signal: ReplaySignal):
        """处理单个信号"""
        self._signals_processed += 1
        self._current_timestamp = signal.timestamp

        # 更新日期和重置日交易计数
        if self._current_date != signal.timestamp.date():
            self._current_date = signal.timestamp.date()
            self._daily_trades = 0

        # 更新价格
        if signal.price_at_signal:
            self._current_prices[signal.symbol] = signal.price_at_signal
        elif self._prices is not None:
            price = self._get_price(signal.symbol, signal.timestamp)
            if price:
                self._current_prices[signal.symbol] = price

        # 更新持仓价格
        self._update_positions()

        # 回调
        if self._on_signal:
            self._on_signal(signal)

        # 验证信号
        if not self._validate_signal(signal):
            self._signals_rejected += 1
            return

        # 检查日交易限制
        if self._daily_trades >= self.config.max_daily_trades:
            self._signals_rejected += 1
            logger.debug(f"Daily trade limit reached, rejecting signal for {signal.symbol}")
            return

        # 执行信号
        trade = await self._execute_signal(signal)

        if trade:
            self._signals_executed += 1
            self._daily_trades += 1

            # 记录权益曲线
            self._record_equity(signal.timestamp)

    def _validate_signal(self, signal: ReplaySignal) -> bool:
        """验证信号"""
        # 强度检查
        if signal.strength < self.config.min_signal_strength:
            return False

        # 置信度检查
        if signal.confidence < self.config.min_signal_confidence:
            return False

        # 方向检查（中性信号忽略）
        if signal.direction == 'neutral':
            return False

        return True

    async def _execute_signal(self, signal: ReplaySignal) -> Optional[ReplayTrade]:
        """执行信号"""
        current_price = self._current_prices.get(signal.symbol)
        if not current_price:
            logger.warning(f"No price available for {signal.symbol}")
            return None

        # 确定交易方向
        if signal.direction in ['strong_long', 'long']:
            side = 'buy'
            position_side = 'long'
        elif signal.direction in ['strong_short', 'short']:
            side = 'sell'
            position_side = 'short'
        else:
            return None

        # 检查现有持仓
        existing_position = self._positions.get(signal.symbol)

        if existing_position:
            # 如果方向相反，先平仓
            if (existing_position.side == 'long' and side == 'sell') or \
               (existing_position.side == 'short' and side == 'buy'):
                await self._close_position(signal.symbol, signal.timestamp, current_price)
            else:
                # 同方向不加仓
                return None

        # 计算仓位大小
        position_value = self._capital * self.config.max_position_weight * signal.strength
        quantity = position_value / current_price

        # 计算滑点和手续费
        slippage = current_price * self.config.slippage_rate
        if side == 'buy':
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage

        commission = position_value * self.config.commission_rate

        # 创建交易
        self._trade_counter += 1
        trade = ReplayTrade(
            trade_id=f"RT-{self._trade_counter:06d}",
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage * quantity,
            signal=signal,
        )

        # 扣除成本
        self._capital -= commission

        # 创建持仓
        stop_loss = None
        take_profit = None

        if self.config.stop_loss_pct > 0:
            if position_side == 'long':
                stop_loss = execution_price * (1 - self.config.stop_loss_pct)
            else:
                stop_loss = execution_price * (1 + self.config.stop_loss_pct)

        if self.config.take_profit_pct > 0:
            if position_side == 'long':
                take_profit = execution_price * (1 + self.config.take_profit_pct)
            else:
                take_profit = execution_price * (1 - self.config.take_profit_pct)

        self._positions[signal.symbol] = ReplayPosition(
            symbol=signal.symbol,
            side=position_side,
            quantity=quantity,
            entry_price=execution_price,
            entry_time=signal.timestamp,
            current_price=execution_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self._trades.append(trade)

        if self._on_trade:
            self._on_trade(trade)

        logger.debug(f"Executed {side} {quantity:.2f} {signal.symbol} @ {execution_price:.4f}")

        return trade

    async def _close_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
    ) -> Optional[ReplayTrade]:
        """平仓"""
        position = self._positions.get(symbol)
        if not position:
            return None

        # 计算滑点
        slippage = price * self.config.slippage_rate
        if position.side == 'long':
            execution_price = price - slippage
            side = 'sell'
        else:
            execution_price = price + slippage
            side = 'buy'

        # 计算盈亏
        if position.side == 'long':
            pnl = (execution_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - execution_price) * position.quantity

        commission = abs(pnl) * self.config.commission_rate if pnl > 0 else position.entry_price * position.quantity * self.config.commission_rate
        pnl -= commission

        pnl_pct = pnl / (position.entry_price * position.quantity)

        # 创建平仓交易
        self._trade_counter += 1
        trade = ReplayTrade(
            trade_id=f"RT-{self._trade_counter:06d}",
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=position.quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage * position.quantity,
            exit_timestamp=timestamp,
            exit_price=execution_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

        # 更新资金
        self._capital += pnl

        # 更新开仓交易的平仓信息
        for t in reversed(self._trades):
            if t.symbol == symbol and t.exit_timestamp is None and t.side != side:
                t.exit_timestamp = timestamp
                t.exit_price = execution_price
                t.pnl = pnl
                t.pnl_pct = pnl_pct
                break

        self._trades.append(trade)
        del self._positions[symbol]

        if self._on_trade:
            self._on_trade(trade)

        logger.debug(f"Closed {symbol} position: PnL={pnl:.2f} ({pnl_pct*100:.2f}%)")

        return trade

    async def _close_all_positions(self):
        """平掉所有持仓"""
        symbols = list(self._positions.keys())
        for symbol in symbols:
            price = self._current_prices.get(symbol, self._positions[symbol].current_price)
            await self._close_position(
                symbol,
                self._current_timestamp or datetime.now(),
                price,
            )

    def _update_positions(self):
        """更新所有持仓的当前价格"""
        for symbol, position in self._positions.items():
            price = self._current_prices.get(symbol)
            if price:
                position.update_price(price)

                if self._on_position_update:
                    self._on_position_update(position)

    def _get_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """获取指定时间的价格"""
        if self._prices is None:
            return None

        try:
            if isinstance(self._prices.index, pd.MultiIndex):
                return self._prices.loc[(timestamp, symbol), 'close']
            else:
                # 获取最近的价格
                idx = self._prices.index.get_indexer([timestamp], method='ffill')[0]
                if idx >= 0:
                    return self._prices.iloc[idx]['close']
        except (KeyError, IndexError):
            pass

        return None

    def _record_equity(self, timestamp: datetime):
        """记录权益曲线点"""
        # 计算总权益
        total_equity = self._capital
        for position in self._positions.values():
            if position.side == 'long':
                total_equity += position.current_price * position.quantity
            else:
                # 空头需要计算保证金占用和浮动盈亏
                total_equity += position.unrealized_pnl

        self._equity_curve.append({
            'timestamp': timestamp.isoformat(),
            'equity': total_equity,
            'cash': self._capital,
            'positions_value': total_equity - self._capital,
            'positions_count': len(self._positions),
        })

    async def _apply_speed_delay(self, timestamp: datetime):
        """应用速度延迟"""
        if self.config.speed == ReplaySpeed.INSTANT:
            return

        # 计算与上一个时间戳的间隔
        if hasattr(self, '_last_timestamp') and self._last_timestamp:
            delta = (timestamp - self._last_timestamp).total_seconds()

            if self.config.speed == ReplaySpeed.REALTIME:
                delay = delta
            elif self.config.speed == ReplaySpeed.FAST:
                delay = delta / 10
            elif self.config.speed == ReplaySpeed.SLOW:
                delay = delta * 2
            else:
                delay = 0

            if delay > 0:
                await asyncio.sleep(min(delay, 1.0))  # 最多等1秒

        self._last_timestamp = timestamp

    def _calculate_result(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> ReplayResult:
        """计算回放结果"""
        # 基本统计
        total_trades = len([t for t in self._trades if t.pnl != 0 or t.exit_timestamp])
        winning_trades = len([t for t in self._trades if t.pnl > 0])
        losing_trades = len([t for t in self._trades if t.pnl < 0])

        total_pnl = sum(t.pnl for t in self._trades)
        total_commission = sum(t.commission for t in self._trades)

        # 绩效指标
        total_return = (self._capital - self.config.initial_capital) / self.config.initial_capital

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 计算夏普比率和最大回撤
        sharpe_ratio = 0.0
        max_drawdown = 0.0

        if self._equity_curve:
            equities = [e['equity'] for e in self._equity_curve]

            if len(equities) > 1:
                returns = pd.Series(equities).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

                peak = equities[0]
                for equity in equities:
                    if equity > peak:
                        peak = equity
                    drawdown = (peak - equity) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

        return ReplayResult(
            mode=self.config.mode,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            initial_capital=self.config.initial_capital,
            final_capital=self._capital,
            equity_curve=self._equity_curve,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_commission=total_commission,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=self._trades,
            signals_processed=self._signals_processed,
            signals_executed=self._signals_executed,
            signals_rejected=self._signals_rejected,
        )

    def _create_empty_result(self, run_start: datetime) -> ReplayResult:
        """创建空结果"""
        return ReplayResult(
            mode=self.config.mode,
            start_time=run_start,
            end_time=datetime.now(),
            duration_seconds=0,
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
        )

    # ============ 便捷方法 ============

    def get_positions(self) -> Dict[str, ReplayPosition]:
        """获取当前持仓"""
        return self._positions.copy()

    def get_equity(self) -> float:
        """获取当前权益"""
        total = self._capital
        for position in self._positions.values():
            if position.side == 'long':
                total += position.current_price * position.quantity
            else:
                total += position.unrealized_pnl
        return total

    def get_trades(self) -> List[ReplayTrade]:
        """获取交易记录"""
        return self._trades.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """获取当前统计"""
        return {
            'is_running': self._is_running,
            'capital': self._capital,
            'equity': self.get_equity(),
            'positions_count': len(self._positions),
            'trades_count': len(self._trades),
            'signals_processed': self._signals_processed,
            'signals_executed': self._signals_executed,
            'signals_rejected': self._signals_rejected,
            'daily_trades': self._daily_trades,
        }


# ============ 工厂函数 ============

def create_strategy_replayer(
    mode: str = 'backtest',
    initial_capital: float = 1_000_000,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    **kwargs,
) -> StrategyReplayer:
    """
    创建策略回放器

    Args:
        mode: 回放模式 (backtest/paper/validation/walk_forward)
        initial_capital: 初始资金
        commission_rate: 手续费率
        slippage_rate: 滑点率
        **kwargs: 其他配置参数

    Returns:
        StrategyReplayer实例
    """
    mode_map = {
        'backtest': ReplayMode.BACKTEST,
        'paper': ReplayMode.PAPER,
        'validation': ReplayMode.VALIDATION,
        'walk_forward': ReplayMode.WALK_FORWARD,
    }

    config = ReplayConfig(
        mode=mode_map.get(mode, ReplayMode.BACKTEST),
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        min_signal_strength=kwargs.get('min_signal_strength', 0.3),
        min_signal_confidence=kwargs.get('min_signal_confidence', 0.5),
        max_position_weight=kwargs.get('max_position_weight', 0.20),
        max_daily_trades=kwargs.get('max_daily_trades', 50),
        stop_loss_pct=kwargs.get('stop_loss_pct', 0.05),
        take_profit_pct=kwargs.get('take_profit_pct', 0.10),
    )

    return StrategyReplayer(config)


async def replay_signals(
    signals: List[Dict[str, Any]],
    prices: Optional[pd.DataFrame] = None,
    config: Optional[ReplayConfig] = None,
) -> ReplayResult:
    """
    快速回放信号

    Args:
        signals: 信号列表
        prices: 价格数据（可选）
        config: 回放配置（可选）

    Returns:
        ReplayResult
    """
    replayer = StrategyReplayer(config)
    replayer.load_signals(signals)

    if prices is not None:
        replayer.load_prices(prices)

    return await replayer.run()
