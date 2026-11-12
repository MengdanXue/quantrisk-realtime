"""
信号桥接器
Signal Bridge

连接信号聚合系统与执行器，实现完整的信号到订单链路:
- 将AggregatedSignal转换为TradingSignal
- 自动化信号处理管道
- 实时监控信号流
- 信号过滤与增强

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class BridgeState(str, Enum):
    """桥接器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class SignalFilterResult(str, Enum):
    """信号过滤结果"""
    PASS = "pass"
    FILTERED_WEAK = "filtered_weak"
    FILTERED_CONFLICT = "filtered_conflict"
    FILTERED_COOLDOWN = "filtered_cooldown"
    FILTERED_POSITION = "filtered_position"
    FILTERED_CUSTOM = "filtered_custom"


# ============ 数据类 ============

@dataclass
class BridgeConfig:
    """桥接器配置"""
    # 信号过滤
    min_fused_strength: float = 0.4        # 最小融合信号强度
    min_fused_confidence: float = 0.5      # 最小融合信号置信度
    min_agreement_ratio: float = 0.5       # 最小信号一致性比率
    max_conflict_level: float = 0.7        # 最大允许冲突水平

    # 冷却机制
    signal_cooldown_seconds: float = 300   # 同标的信号冷却时间(秒)
    trade_cooldown_seconds: float = 60     # 交易后冷却时间(秒)

    # 信号增强
    use_sentiment_boost: bool = True       # 是否使用舆情增强
    sentiment_boost_factor: float = 1.2    # 舆情增强系数
    extreme_sentiment_threshold: float = 0.8  # 极端舆情阈值

    # 执行控制
    auto_execute: bool = True              # 自动执行订单
    require_confirmation: bool = False     # 需要确认
    max_signals_per_minute: int = 10       # 每分钟最大信号数

    # 监控
    log_all_signals: bool = True           # 记录所有信号
    log_filtered_signals: bool = True      # 记录过滤的信号


@dataclass
class BridgedSignal:
    """桥接后的信号"""
    signal_id: str
    original_signal: Any  # AggregatedSignal
    trading_signal: Any   # TradingSignal
    filter_result: SignalFilterResult
    filter_reason: str = ""
    processed_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'symbol': self.trading_signal.symbol if self.trading_signal else None,
            'filter_result': self.filter_result.value,
            'filter_reason': self.filter_reason,
            'processed_at': self.processed_at.isoformat(),
            'executed': self.executed,
        }


@dataclass
class BridgeStatistics:
    """桥接器统计"""
    total_signals_received: int = 0
    signals_passed: int = 0
    signals_filtered: int = 0
    signals_executed: int = 0
    signals_failed: int = 0

    # 按原因统计过滤
    filtered_by_strength: int = 0
    filtered_by_confidence: int = 0
    filtered_by_conflict: int = 0
    filtered_by_cooldown: int = 0
    filtered_by_position: int = 0

    # 执行统计
    total_buy_signals: int = 0
    total_sell_signals: int = 0
    execution_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_signals_received': self.total_signals_received,
            'signals_passed': self.signals_passed,
            'signals_filtered': self.signals_filtered,
            'signals_executed': self.signals_executed,
            'signals_failed': self.signals_failed,
            'filtered_by_strength': self.filtered_by_strength,
            'filtered_by_confidence': self.filtered_by_confidence,
            'filtered_by_conflict': self.filtered_by_conflict,
            'filtered_by_cooldown': self.filtered_by_cooldown,
            'filtered_by_position': self.filtered_by_position,
            'total_buy_signals': self.total_buy_signals,
            'total_sell_signals': self.total_sell_signals,
            'execution_success_rate': self.execution_success_rate,
        }


# ============ 信号桥接器 ============

class SignalBridge:
    """
    信号桥接器

    连接信号聚合系统(engine/signals)与执行器(realtime/signal_executor)
    """

    def __init__(
        self,
        config: BridgeConfig,
        signal_executor: Optional[Any] = None,
        on_signal_received: Optional[Callable[[Any], Awaitable[None]]] = None,
        on_signal_filtered: Optional[Callable[[BridgedSignal], Awaitable[None]]] = None,
        on_signal_executed: Optional[Callable[[BridgedSignal], Awaitable[None]]] = None,
    ):
        self.config = config
        self.signal_executor = signal_executor

        # 回调
        self.on_signal_received = on_signal_received
        self.on_signal_filtered = on_signal_filtered
        self.on_signal_executed = on_signal_executed

        # 状态
        self._state = BridgeState.IDLE
        self._statistics = BridgeStatistics()

        # 冷却追踪
        self._signal_cooldowns: Dict[str, datetime] = {}  # symbol -> last_signal_time
        self._trade_cooldowns: Dict[str, datetime] = {}   # symbol -> last_trade_time

        # 当前持仓追踪
        self._current_positions: Set[str] = set()  # symbols with open positions

        # 信号历史
        self._signal_history: List[BridgedSignal] = []
        self._max_history = 1000

        # 速率限制
        self._recent_signals: List[datetime] = []

    @property
    def state(self) -> BridgeState:
        return self._state

    @property
    def statistics(self) -> BridgeStatistics:
        return self._statistics

    def start(self):
        """启动桥接器"""
        self._state = BridgeState.RUNNING
        logger.info("SignalBridge started")

    def stop(self):
        """停止桥接器"""
        self._state = BridgeState.IDLE
        logger.info("SignalBridge stopped")

    def pause(self):
        """暂停桥接器"""
        self._state = BridgeState.PAUSED
        logger.info("SignalBridge paused")

    def resume(self):
        """恢复桥接器"""
        if self._state == BridgeState.PAUSED:
            self._state = BridgeState.RUNNING
            logger.info("SignalBridge resumed")

    def set_executor(self, executor: Any):
        """设置执行器"""
        self.signal_executor = executor

    def update_positions(self, positions: Set[str]):
        """更新当前持仓"""
        self._current_positions = positions

    def add_position(self, symbol: str):
        """添加持仓"""
        self._current_positions.add(symbol)

    def remove_position(self, symbol: str):
        """移除持仓"""
        self._current_positions.discard(symbol)

    # ============ 核心处理 ============

    async def process_aggregated_signal(
        self,
        aggregated_signal: Any,
        portfolio_id: Optional[int] = None,
        current_price: Optional[float] = None,
    ) -> BridgedSignal:
        """
        处理聚合信号

        Args:
            aggregated_signal: 来自SignalAggregator的聚合信号
            portfolio_id: 投资组合ID
            current_price: 当前价格

        Returns:
            桥接信号结果
        """
        self._statistics.total_signals_received += 1

        # 触发接收回调
        if self.on_signal_received:
            await self.on_signal_received(aggregated_signal)

        # 检查状态
        if self._state != BridgeState.RUNNING:
            return self._create_filtered_signal(
                aggregated_signal,
                SignalFilterResult.FILTERED_CUSTOM,
                "Bridge not running"
            )

        # 检查速率限制
        if not self._check_rate_limit():
            return self._create_filtered_signal(
                aggregated_signal,
                SignalFilterResult.FILTERED_CUSTOM,
                "Rate limit exceeded"
            )

        # 提取信号信息
        symbol = aggregated_signal.symbol
        fused = aggregated_signal.fused_signal

        # 过滤检查
        filter_result, filter_reason = self._apply_filters(aggregated_signal)

        if filter_result != SignalFilterResult.PASS:
            bridged = self._create_filtered_signal(
                aggregated_signal, filter_result, filter_reason
            )
            if self.on_signal_filtered:
                await self.on_signal_filtered(bridged)
            return bridged

        # 转换为TradingSignal
        trading_signal = self._convert_to_trading_signal(aggregated_signal)

        # 应用信号增强
        if self.config.use_sentiment_boost:
            trading_signal = self._apply_sentiment_boost(
                trading_signal, aggregated_signal
            )

        # 创建桥接信号
        bridged = BridgedSignal(
            signal_id=f"BRG-{uuid.uuid4().hex[:8].upper()}",
            original_signal=aggregated_signal,
            trading_signal=trading_signal,
            filter_result=SignalFilterResult.PASS,
        )

        self._statistics.signals_passed += 1

        # 更新冷却
        self._signal_cooldowns[symbol] = datetime.now()

        # 自动执行
        if self.config.auto_execute and self.signal_executor:
            result = await self._execute_signal(bridged, portfolio_id, current_price)
            bridged.executed = result is not None and result.success
            bridged.execution_result = result

            if bridged.executed:
                self._statistics.signals_executed += 1
                self._trade_cooldowns[symbol] = datetime.now()

                if self.on_signal_executed:
                    await self.on_signal_executed(bridged)
            else:
                self._statistics.signals_failed += 1

        # 记录历史
        self._add_to_history(bridged)

        if self.config.log_all_signals:
            logger.info(f"Signal processed: {symbol} {fused.direction.value} "
                       f"strength={fused.strength:.2f} -> {'executed' if bridged.executed else 'not executed'}")

        return bridged

    def _apply_filters(
        self,
        signal: Any,
    ) -> tuple[SignalFilterResult, str]:
        """应用过滤规则"""
        symbol = signal.symbol
        fused = signal.fused_signal

        # 1. 强度检查
        if fused.strength < self.config.min_fused_strength:
            self._statistics.filtered_by_strength += 1
            self._statistics.signals_filtered += 1
            return (
                SignalFilterResult.FILTERED_WEAK,
                f"Strength {fused.strength:.2f} < {self.config.min_fused_strength}"
            )

        # 2. 置信度检查
        if fused.confidence < self.config.min_fused_confidence:
            self._statistics.filtered_by_confidence += 1
            self._statistics.signals_filtered += 1
            return (
                SignalFilterResult.FILTERED_WEAK,
                f"Confidence {fused.confidence:.2f} < {self.config.min_fused_confidence}"
            )

        # 3. 一致性检查
        if hasattr(fused, 'agreement_ratio'):
            if fused.agreement_ratio < self.config.min_agreement_ratio:
                self._statistics.filtered_by_conflict += 1
                self._statistics.signals_filtered += 1
                return (
                    SignalFilterResult.FILTERED_CONFLICT,
                    f"Agreement ratio {fused.agreement_ratio:.2f} < {self.config.min_agreement_ratio}"
                )

        # 4. 冲突水平检查
        if hasattr(fused, 'conflict_level'):
            if fused.conflict_level > self.config.max_conflict_level:
                self._statistics.filtered_by_conflict += 1
                self._statistics.signals_filtered += 1
                return (
                    SignalFilterResult.FILTERED_CONFLICT,
                    f"Conflict level {fused.conflict_level:.2f} > {self.config.max_conflict_level}"
                )

        # 5. 冷却检查
        if symbol in self._signal_cooldowns:
            elapsed = (datetime.now() - self._signal_cooldowns[symbol]).total_seconds()
            if elapsed < self.config.signal_cooldown_seconds:
                self._statistics.filtered_by_cooldown += 1
                self._statistics.signals_filtered += 1
                return (
                    SignalFilterResult.FILTERED_COOLDOWN,
                    f"Signal cooldown: {elapsed:.0f}s < {self.config.signal_cooldown_seconds}s"
                )

        if symbol in self._trade_cooldowns:
            elapsed = (datetime.now() - self._trade_cooldowns[symbol]).total_seconds()
            if elapsed < self.config.trade_cooldown_seconds:
                self._statistics.filtered_by_cooldown += 1
                self._statistics.signals_filtered += 1
                return (
                    SignalFilterResult.FILTERED_COOLDOWN,
                    f"Trade cooldown: {elapsed:.0f}s < {self.config.trade_cooldown_seconds}s"
                )

        return (SignalFilterResult.PASS, "")

    def _convert_to_trading_signal(self, aggregated_signal: Any) -> Any:
        """将AggregatedSignal转换为TradingSignal"""
        from .signal_executor import TradingSignal, SignalDirection

        fused = aggregated_signal.fused_signal

        # 映射方向
        direction_map = {
            'strong_long': SignalDirection.STRONG_LONG,
            'long': SignalDirection.LONG,
            'neutral': SignalDirection.NEUTRAL,
            'short': SignalDirection.SHORT,
            'strong_short': SignalDirection.STRONG_SHORT,
        }

        direction_value = fused.direction.value if hasattr(fused.direction, 'value') else str(fused.direction)
        signal_direction = direction_map.get(direction_value.lower(), SignalDirection.NEUTRAL)

        # 提取止损止盈(如果策略信号中有)
        stop_loss = None
        take_profit = None
        for strat_signal in aggregated_signal.strategy_signals:
            if hasattr(strat_signal, 'stop_loss') and strat_signal.stop_loss:
                stop_loss = strat_signal.stop_loss
            if hasattr(strat_signal, 'take_profit') and strat_signal.take_profit:
                take_profit = strat_signal.take_profit

        # 构建原因说明
        reasons = []
        for s in aggregated_signal.strategy_signals[:3]:  # 取前3个策略
            if hasattr(s, 'reason') and s.reason:
                reasons.append(f"{s.strategy_name}: {s.reason}")

        return TradingSignal(
            signal_id=f"SIG-{uuid.uuid4().hex[:8].upper()}",
            symbol=aggregated_signal.symbol,
            direction=signal_direction,
            strength=fused.strength,
            confidence=fused.confidence,
            source="signal_aggregator",
            generated_at=aggregated_signal.generated_at,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'market_regime': aggregated_signal.market_regime.value if hasattr(aggregated_signal.market_regime, 'value') else str(aggregated_signal.market_regime),
                'strategy_count': len(aggregated_signal.strategy_signals),
                'indicator_count': len(aggregated_signal.indicator_signals),
                'reasons': reasons,
            }
        )

    def _apply_sentiment_boost(self, signal: Any, aggregated: Any) -> Any:
        """应用舆情增强"""
        if not aggregated.sentiment_signals:
            return signal

        # 检查是否有极端舆情
        for sent in aggregated.sentiment_signals:
            if hasattr(sent, 'strength') and abs(sent.strength) >= self.config.extreme_sentiment_threshold:
                # 检查舆情方向是否与信号一致
                sent_direction = getattr(sent, 'direction', None)
                if sent_direction:
                    sent_is_bullish = 'long' in str(sent_direction).lower() or 'bull' in str(sent_direction).lower()
                    signal_is_bullish = signal.direction.value in ['strong_long', 'long']

                    if sent_is_bullish == signal_is_bullish:
                        # 方向一致，增强信号
                        signal.strength = min(signal.strength * self.config.sentiment_boost_factor, 1.0)
                        signal.confidence = min(signal.confidence * 1.1, 1.0)

                        if hasattr(signal, 'metadata'):
                            signal.metadata['sentiment_boosted'] = True

        return signal

    async def _execute_signal(
        self,
        bridged: BridgedSignal,
        portfolio_id: Optional[int],
        current_price: Optional[float],
    ) -> Optional[Any]:
        """执行信号"""
        if not self.signal_executor:
            return None

        try:
            result = await self.signal_executor._execute_signal(
                signal=bridged.trading_signal,
                portfolio_id=portfolio_id,
                portfolio_value=1000000,  # 默认值，实际应从组合服务获取
                current_price=current_price,
            )

            # 更新买卖统计
            if result and result.success:
                if bridged.trading_signal.direction.value in ['strong_long', 'long']:
                    self._statistics.total_buy_signals += 1
                else:
                    self._statistics.total_sell_signals += 1

            return result

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return None

    def _create_filtered_signal(
        self,
        original: Any,
        result: SignalFilterResult,
        reason: str,
    ) -> BridgedSignal:
        """创建被过滤的信号记录"""
        bridged = BridgedSignal(
            signal_id=f"BRG-{uuid.uuid4().hex[:8].upper()}",
            original_signal=original,
            trading_signal=None,
            filter_result=result,
            filter_reason=reason,
        )

        self._add_to_history(bridged)

        if self.config.log_filtered_signals:
            logger.debug(f"Signal filtered: {original.symbol} - {result.value}: {reason}")

        return bridged

    def _check_rate_limit(self) -> bool:
        """检查速率限制"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # 清理旧记录
        self._recent_signals = [t for t in self._recent_signals if t > cutoff]

        if len(self._recent_signals) >= self.config.max_signals_per_minute:
            return False

        self._recent_signals.append(now)
        return True

    def _add_to_history(self, bridged: BridgedSignal):
        """添加到历史"""
        self._signal_history.append(bridged)
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]

    # ============ 批量处理 ============

    async def process_signals_batch(
        self,
        signals: List[Any],
        portfolio_id: Optional[int] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[BridgedSignal]:
        """
        批量处理信号

        Args:
            signals: 聚合信号列表
            portfolio_id: 投资组合ID
            prices: 价格字典 {symbol: price}

        Returns:
            桥接信号结果列表
        """
        results = []
        prices = prices or {}

        for signal in signals:
            price = prices.get(signal.symbol)
            result = await self.process_aggregated_signal(
                signal, portfolio_id, price
            )
            results.append(result)

        return results

    # ============ 查询方法 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._statistics.to_dict()

        # 计算成功率
        total_executed = self._statistics.signals_executed + self._statistics.signals_failed
        if total_executed > 0:
            stats['execution_success_rate'] = self._statistics.signals_executed / total_executed
        else:
            stats['execution_success_rate'] = 0.0

        stats['state'] = self._state.value
        stats['current_positions'] = list(self._current_positions)
        stats['active_cooldowns'] = len(self._signal_cooldowns)

        return stats

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        passed_only: bool = False,
        limit: int = 100,
    ) -> List[BridgedSignal]:
        """获取信号历史"""
        history = self._signal_history

        if symbol:
            history = [s for s in history if s.original_signal.symbol == symbol]

        if passed_only:
            history = [s for s in history if s.filter_result == SignalFilterResult.PASS]

        return history[-limit:]

    def clear_cooldowns(self, symbol: Optional[str] = None):
        """清除冷却"""
        if symbol:
            self._signal_cooldowns.pop(symbol, None)
            self._trade_cooldowns.pop(symbol, None)
        else:
            self._signal_cooldowns.clear()
            self._trade_cooldowns.clear()


# ============ 工厂函数 ============

def create_signal_bridge(
    executor: Optional[Any] = None,
    min_strength: float = 0.4,
    min_confidence: float = 0.5,
    auto_execute: bool = True,
    **kwargs,
) -> SignalBridge:
    """
    创建信号桥接器

    Args:
        executor: 信号执行器
        min_strength: 最小信号强度
        min_confidence: 最小置信度
        auto_execute: 自动执行
        **kwargs: 其他配置参数

    Returns:
        SignalBridge实例
    """
    config = BridgeConfig(
        min_fused_strength=min_strength,
        min_fused_confidence=min_confidence,
        auto_execute=auto_execute,
        **{k: v for k, v in kwargs.items() if hasattr(BridgeConfig, k)},
    )

    return SignalBridge(config=config, signal_executor=executor)
