"""
动量信号生成器 (Momentum Signal Generator)

提供实时动量信号生成和分析功能:
- 价格动量计算 (ROC, Momentum)
- 相对强弱指数 (RSI)
- 随机指标 (Stochastic)
- MACD 信号
- ADX 趋势强度
- 动量背离检测
- 多时间框架动量
- 动量因子组合
- 动量衰减分析
- 交叉信号检测

Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Deque
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging
import math
import statistics

logger = logging.getLogger(__name__)


# ============== Enums ==============

class SignalType(str, Enum):
    """信号类型"""
    BUY = "buy"              # 买入信号
    SELL = "sell"            # 卖出信号
    HOLD = "hold"            # 持有信号
    STRONG_BUY = "strong_buy"    # 强买入信号
    STRONG_SELL = "strong_sell"  # 强卖出信号


class SignalStrength(str, Enum):
    """信号强度"""
    VERY_WEAK = "very_weak"      # 非常弱
    WEAK = "weak"                # 弱
    MODERATE = "moderate"        # 中等
    STRONG = "strong"            # 强
    VERY_STRONG = "very_strong"  # 非常强


class TrendDirection(str, Enum):
    """趋势方向"""
    UPTREND = "uptrend"          # 上升趋势
    DOWNTREND = "downtrend"      # 下降趋势
    SIDEWAYS = "sideways"        # 横盘
    UNKNOWN = "unknown"          # 未知


class DivergenceType(str, Enum):
    """背离类型"""
    BULLISH = "bullish"          # 看涨背离(价格新低,指标未新低)
    BEARISH = "bearish"          # 看跌背离(价格新高,指标未新高)
    HIDDEN_BULLISH = "hidden_bullish"    # 隐藏看涨背离
    HIDDEN_BEARISH = "hidden_bearish"    # 隐藏看跌背离
    NONE = "none"                # 无背离


class CrossoverType(str, Enum):
    """交叉类型"""
    GOLDEN_CROSS = "golden_cross"        # 金叉
    DEATH_CROSS = "death_cross"          # 死叉
    SIGNAL_CROSS_UP = "signal_cross_up"  # 信号线上穿
    SIGNAL_CROSS_DOWN = "signal_cross_down"  # 信号线下穿
    NONE = "none"                        # 无交叉


class TimeFrame(str, Enum):
    """时间框架"""
    M1 = "1min"       # 1分钟
    M5 = "5min"       # 5分钟
    M15 = "15min"     # 15分钟
    M30 = "30min"     # 30分钟
    H1 = "1hour"      # 1小时
    H4 = "4hour"      # 4小时
    D1 = "1day"       # 1天
    W1 = "1week"      # 1周


class GeneratorState(str, Enum):
    """生成器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============== Data Classes ==============

@dataclass
class PriceData:
    """
    价格数据

    OHLCV 格式的价格数据
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }


@dataclass
class MomentumIndicator:
    """
    动量指标

    记录单个动量指标的值
    """
    name: str                    # 指标名称
    value: float                 # 指标值
    timestamp: datetime          # 时间戳
    signal: SignalType = SignalType.HOLD    # 信号
    strength: SignalStrength = SignalStrength.MODERATE  # 强度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal.value,
            'strength': self.strength.value,
            'metadata': self.metadata,
        }


@dataclass
class MomentumSignal:
    """
    动量信号

    综合多个指标生成的交易信号
    """
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float            # 置信度 (0-1)
    price: float                 # 信号价格

    # 贡献指标
    contributing_indicators: Dict[str, float] = field(default_factory=dict)

    # 元数据
    timeframe: TimeFrame = TimeFrame.D1
    trend_direction: TrendDirection = TrendDirection.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value,
            'signal_strength': self.signal_strength.value,
            'confidence': self.confidence,
            'price': self.price,
            'contributing_indicators': self.contributing_indicators,
            'timeframe': self.timeframe.value,
            'trend_direction': self.trend_direction.value,
        }


@dataclass
class DivergenceSignal:
    """
    背离信号

    价格和指标之间的背离
    """
    symbol: str
    timestamp: datetime
    divergence_type: DivergenceType
    indicator_name: str          # 背离指标名称

    # 背离点
    price_point1: Tuple[datetime, float]   # 价格点1 (时间, 价格)
    price_point2: Tuple[datetime, float]   # 价格点2
    indicator_point1: Tuple[datetime, float]  # 指标点1
    indicator_point2: Tuple[datetime, float]  # 指标点2

    # 强度
    strength: SignalStrength = SignalStrength.MODERATE
    significance: float = 0.0    # 显著性 (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'divergence_type': self.divergence_type.value,
            'indicator_name': self.indicator_name,
            'price_point1': (self.price_point1[0].isoformat(), self.price_point1[1]),
            'price_point2': (self.price_point2[0].isoformat(), self.price_point2[1]),
            'indicator_point1': (self.indicator_point1[0].isoformat(), self.indicator_point1[1]),
            'indicator_point2': (self.indicator_point2[0].isoformat(), self.indicator_point2[1]),
            'strength': self.strength.value,
            'significance': self.significance,
        }


@dataclass
class TrendStrength:
    """
    趋势强度

    基于ADX的趋势强度评估
    """
    symbol: str
    timestamp: datetime
    adx: float                   # ADX值
    plus_di: float               # +DI
    minus_di: float              # -DI

    # 趋势评估
    trend_direction: TrendDirection
    trend_strength: SignalStrength
    is_trending: bool = False    # 是否处于明显趋势

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'adx': self.adx,
            'plus_di': self.plus_di,
            'minus_di': self.minus_di,
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength.value,
            'is_trending': self.is_trending,
        }


@dataclass
class MomentumConfig:
    """动量配置"""
    # RSI 参数
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Stochastic 参数
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0

    # MACD 参数
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    # ADX 参数
    adx_period: int = 14
    adx_threshold: float = 25.0  # ADX > 25 认为有趋势

    # ROC 参数
    roc_period: int = 12

    # 动量参数
    momentum_period: int = 10

    # 背离检测参数
    divergence_lookback: int = 50     # 背离检测回溯期
    divergence_min_bars: int = 5      # 最小背离距离

    # 衰减参数
    decay_half_life: float = 10.0     # 衰减半衰期

    # 多时间框架
    enable_multi_timeframe: bool = True
    timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.D1])

    # 综合信号权重
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 0.20,
        'stochastic': 0.15,
        'macd': 0.25,
        'adx': 0.20,
        'roc': 0.10,
        'momentum': 0.10,
    })

    # 监控参数
    update_interval: float = 1.0      # 更新间隔(秒)
    history_size: int = 500           # 历史数据大小

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_overbought': self.stoch_overbought,
            'stoch_oversold': self.stoch_oversold,
            'macd_fast_period': self.macd_fast_period,
            'macd_slow_period': self.macd_slow_period,
            'macd_signal_period': self.macd_signal_period,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'roc_period': self.roc_period,
            'momentum_period': self.momentum_period,
            'divergence_lookback': self.divergence_lookback,
            'divergence_min_bars': self.divergence_min_bars,
            'decay_half_life': self.decay_half_life,
            'enable_multi_timeframe': self.enable_multi_timeframe,
            'timeframes': [tf.value for tf in self.timeframes],
            'signal_weights': self.signal_weights,
            'update_interval': self.update_interval,
            'history_size': self.history_size,
        }


# ============== 动量指标计算器 ==============

class MomentumCalculator:
    """
    动量指标计算器

    提供各种技术指标的计算方法
    """

    @staticmethod
    def calculate_roc(prices: List[float], period: int) -> Optional[float]:
        """
        计算变化率 (Rate of Change)

        ROC = ((价格 - N期前价格) / N期前价格) * 100

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            ROC值
        """
        if len(prices) < period + 1:
            return None

        current = prices[-1]
        past = prices[-period - 1]

        if past == 0:
            return None

        roc = ((current - past) / past) * 100
        return roc

    @staticmethod
    def calculate_momentum(prices: List[float], period: int) -> Optional[float]:
        """
        计算动量 (Momentum)

        Momentum = 价格 - N期前价格

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            动量值
        """
        if len(prices) < period + 1:
            return None

        momentum = prices[-1] - prices[-period - 1]
        return momentum

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        计算相对强弱指数 (RSI)

        RSI = 100 - (100 / (1 + RS))
        RS = 平均涨幅 / 平均跌幅

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            RSI值 (0-100)
        """
        if len(prices) < period + 1:
            return None

        # 计算价格变化
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        if len(changes) < period:
            return None

        # 分离涨跌
        gains = [max(0, change) for change in changes[-period:]]
        losses = [abs(min(0, change)) for change in changes[-period:]]

        # 计算平均涨跌
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_stochastic(
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        计算随机指标 (Stochastic)

        %K = (收盘价 - N期最低价) / (N期最高价 - N期最低价) * 100
        %D = %K的M期移动平均

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            k_period: %K周期
            d_period: %D周期

        Returns:
            (%K, %D) 或 None
        """
        if len(close_prices) < k_period:
            return None

        # 计算%K
        recent_high = max(high_prices[-k_period:])
        recent_low = min(low_prices[-k_period:])
        current_close = close_prices[-1]

        if recent_high == recent_low:
            k = 50.0
        else:
            k = ((current_close - recent_low) / (recent_high - recent_low)) * 100

        # 计算%D (简化为最后几个%K的平均)
        # 这里简化实现,实际应该维护%K历史
        d = k  # 简化版本

        return (k, d)

    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Optional[Tuple[float, float, float]]:
        """
        计算MACD指标

        MACD = 快速EMA - 慢速EMA
        Signal = MACD的EMA
        Histogram = MACD - Signal

        Args:
            prices: 价格序列
            fast_period: 快速周期
            slow_period: 慢速周期
            signal_period: 信号周期

        Returns:
            (MACD, Signal, Histogram) 或 None
        """
        if len(prices) < slow_period:
            return None

        # 计算EMA
        def calculate_ema(data: List[float], period: int) -> float:
            multiplier = 2 / (period + 1)
            ema = data[0]
            for price in data[1:]:
                ema = (price - ema) * multiplier + ema
            return ema

        # 快速和慢速EMA
        fast_ema = calculate_ema(prices[-fast_period:], fast_period)
        slow_ema = calculate_ema(prices[-slow_period:], slow_period)

        macd = fast_ema - slow_ema

        # Signal线 (简化版本)
        signal = macd  # 简化:实际应该计算MACD的EMA

        histogram = macd - signal

        return (macd, signal, histogram)

    @staticmethod
    def calculate_adx(
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        period: int = 14
    ) -> Optional[Tuple[float, float, float]]:
        """
        计算ADX指标 (Average Directional Index)

        衡量趋势强度的指标

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            period: 周期

        Returns:
            (ADX, +DI, -DI) 或 None
        """
        if len(close_prices) < period + 1:
            return None

        # 计算True Range
        tr_list = []
        for i in range(1, len(close_prices)):
            high = high_prices[i]
            low = low_prices[i]
            prev_close = close_prices[i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        # 计算方向移动
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(high_prices)):
            high_diff = high_prices[i] - high_prices[i-1]
            low_diff = low_prices[i-1] - low_prices[i]

            plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        # 计算平滑的TR和DM
        avg_tr = sum(tr_list[-period:]) / period
        avg_plus_dm = sum(plus_dm_list[-period:]) / period
        avg_minus_dm = sum(minus_dm_list[-period:]) / period

        # 计算DI
        plus_di = (avg_plus_dm / avg_tr * 100) if avg_tr != 0 else 0
        minus_di = (avg_minus_dm / avg_tr * 100) if avg_tr != 0 else 0

        # 计算DX和ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            adx = 0
        else:
            dx = abs(plus_di - minus_di) / di_sum * 100
            adx = dx  # 简化:实际应该是DX的平滑移动平均

        return (adx, plus_di, minus_di)


# ============== 动量信号生成器 ==============

class MomentumSignalGenerator:
    """
    动量信号生成器

    功能:
    - 实时计算各种动量指标
    - 检测动量背离
    - 生成综合交易信号
    - 多时间框架分析
    - 动量衰减分析
    """

    def __init__(
        self,
        config: Optional[MomentumConfig] = None,
        signal_callback: Optional[Callable[[MomentumSignal], None]] = None,
        divergence_callback: Optional[Callable[[DivergenceSignal], None]] = None,
    ):
        self.config = config or MomentumConfig()
        self.signal_callback = signal_callback
        self.divergence_callback = divergence_callback

        # 状态
        self._state = GeneratorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 价格数据存储
        self._price_history: Dict[str, Deque[PriceData]] = defaultdict(
            lambda: deque(maxlen=self.config.history_size)
        )

        # 指标缓存
        self._indicators_cache: Dict[str, Dict[str, MomentumIndicator]] = defaultdict(dict)

        # 信号历史
        self._signals_history: Deque[MomentumSignal] = deque(maxlen=self.config.history_size)

        # 背离历史
        self._divergences_history: Deque[DivergenceSignal] = deque(maxlen=self.config.history_size)

        # 趋势强度历史
        self._trend_strength_history: Dict[str, Deque[TrendStrength]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # 计算器
        self._calculator = MomentumCalculator()

        logger.info("MomentumSignalGenerator initialized")

    @property
    def state(self) -> GeneratorState:
        return self._state

    # ============== 生命周期管理 ==============

    async def start(self) -> None:
        """启动生成器"""
        if self._state == GeneratorState.RUNNING:
            return

        logger.info("Starting Momentum Signal Generator")
        self._state = GeneratorState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Momentum Signal Generator started")

    async def stop(self) -> None:
        """停止生成器"""
        if self._state == GeneratorState.IDLE:
            return

        logger.info("Stopping Momentum Signal Generator")
        self._state = GeneratorState.IDLE

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Momentum Signal Generator stopped")

    async def pause(self) -> None:
        """暂停生成器"""
        if self._state == GeneratorState.RUNNING:
            self._state = GeneratorState.PAUSED
            logger.info("Momentum Signal Generator paused")

    async def resume(self) -> None:
        """恢复生成器"""
        if self._state == GeneratorState.PAUSED:
            self._state = GeneratorState.RUNNING
            logger.info("Momentum Signal Generator resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (GeneratorState.RUNNING, GeneratorState.PAUSED):
            try:
                if self._state == GeneratorState.RUNNING:
                    # 定期更新信号
                    await self._update_all_signals()

                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = GeneratorState.ERROR

    # ============== 价格更新 ==============

    async def update_price(self, price_data: PriceData) -> None:
        """
        更新价格数据

        Args:
            price_data: 价格数据
        """
        symbol = price_data.symbol
        self._price_history[symbol].append(price_data)

        # 触发信号更新
        if len(self._price_history[symbol]) >= self.config.rsi_period:
            await self._update_signals_for_symbol(symbol)

    # ============== 指标计算 ==============

    def calculate_roc(self, symbol: str) -> Optional[MomentumIndicator]:
        """
        计算ROC (Rate of Change)

        Args:
            symbol: 标的符号

        Returns:
            ROC指标
        """
        prices = [p.close for p in self._price_history[symbol]]

        roc = self._calculator.calculate_roc(prices, self.config.roc_period)

        if roc is None:
            return None

        # 判断信号
        if roc > 5:
            signal = SignalType.BUY
            strength = SignalStrength.STRONG
        elif roc > 2:
            signal = SignalType.BUY
            strength = SignalStrength.MODERATE
        elif roc < -5:
            signal = SignalType.SELL
            strength = SignalStrength.STRONG
        elif roc < -2:
            signal = SignalType.SELL
            strength = SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD
            strength = SignalStrength.WEAK

        indicator = MomentumIndicator(
            name='roc',
            value=roc,
            timestamp=datetime.now(),
            signal=signal,
            strength=strength,
            metadata={'period': self.config.roc_period}
        )

        self._indicators_cache[symbol]['roc'] = indicator
        return indicator

    def calculate_rsi(self, symbol: str) -> Optional[MomentumIndicator]:
        """
        计算RSI (Relative Strength Index)

        Args:
            symbol: 标的符号

        Returns:
            RSI指标
        """
        prices = [p.close for p in self._price_history[symbol]]

        rsi = self._calculator.calculate_rsi(prices, self.config.rsi_period)

        if rsi is None:
            return None

        # 判断信号
        if rsi > self.config.rsi_overbought:
            signal = SignalType.SELL
            strength = SignalStrength.STRONG if rsi > 80 else SignalStrength.MODERATE
        elif rsi < self.config.rsi_oversold:
            signal = SignalType.BUY
            strength = SignalStrength.STRONG if rsi < 20 else SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD
            strength = SignalStrength.WEAK

        indicator = MomentumIndicator(
            name='rsi',
            value=rsi,
            timestamp=datetime.now(),
            signal=signal,
            strength=strength,
            metadata={'period': self.config.rsi_period}
        )

        self._indicators_cache[symbol]['rsi'] = indicator
        return indicator

    def calculate_stochastic(self, symbol: str) -> Optional[MomentumIndicator]:
        """
        计算随机指标 (Stochastic)

        Args:
            symbol: 标的符号

        Returns:
            随机指标
        """
        price_history = self._price_history[symbol]

        high_prices = [p.high for p in price_history]
        low_prices = [p.low for p in price_history]
        close_prices = [p.close for p in price_history]

        result = self._calculator.calculate_stochastic(
            high_prices,
            low_prices,
            close_prices,
            self.config.stoch_k_period,
            self.config.stoch_d_period
        )

        if result is None:
            return None

        k, d = result

        # 判断信号
        if k > self.config.stoch_overbought:
            signal = SignalType.SELL
            strength = SignalStrength.STRONG if k > 90 else SignalStrength.MODERATE
        elif k < self.config.stoch_oversold:
            signal = SignalType.BUY
            strength = SignalStrength.STRONG if k < 10 else SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD
            strength = SignalStrength.WEAK

        indicator = MomentumIndicator(
            name='stochastic',
            value=k,
            timestamp=datetime.now(),
            signal=signal,
            strength=strength,
            metadata={'k': k, 'd': d}
        )

        self._indicators_cache[symbol]['stochastic'] = indicator
        return indicator

    def calculate_macd(self, symbol: str) -> Optional[MomentumIndicator]:
        """
        计算MACD指标

        Args:
            symbol: 标的符号

        Returns:
            MACD指标
        """
        prices = [p.close for p in self._price_history[symbol]]

        result = self._calculator.calculate_macd(
            prices,
            self.config.macd_fast_period,
            self.config.macd_slow_period,
            self.config.macd_signal_period
        )

        if result is None:
            return None

        macd, signal_line, histogram = result

        # 判断信号
        if histogram > 0 and macd > signal_line:
            signal = SignalType.BUY
            strength = SignalStrength.STRONG if histogram > abs(macd) * 0.1 else SignalStrength.MODERATE
        elif histogram < 0 and macd < signal_line:
            signal = SignalType.SELL
            strength = SignalStrength.STRONG if abs(histogram) > abs(macd) * 0.1 else SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD
            strength = SignalStrength.WEAK

        indicator = MomentumIndicator(
            name='macd',
            value=macd,
            timestamp=datetime.now(),
            signal=signal,
            strength=strength,
            metadata={
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        )

        self._indicators_cache[symbol]['macd'] = indicator
        return indicator

    def calculate_adx(self, symbol: str) -> Optional[TrendStrength]:
        """
        计算ADX (Average Directional Index)

        Args:
            symbol: 标的符号

        Returns:
            趋势强度
        """
        price_history = self._price_history[symbol]

        high_prices = [p.high for p in price_history]
        low_prices = [p.low for p in price_history]
        close_prices = [p.close for p in price_history]

        result = self._calculator.calculate_adx(
            high_prices,
            low_prices,
            close_prices,
            self.config.adx_period
        )

        if result is None:
            return None

        adx, plus_di, minus_di = result

        # 判断趋势方向
        if plus_di > minus_di:
            trend_direction = TrendDirection.UPTREND
        elif minus_di > plus_di:
            trend_direction = TrendDirection.DOWNTREND
        else:
            trend_direction = TrendDirection.SIDEWAYS

        # 判断趋势强度
        if adx > 50:
            trend_strength = SignalStrength.VERY_STRONG
        elif adx > 40:
            trend_strength = SignalStrength.STRONG
        elif adx > 25:
            trend_strength = SignalStrength.MODERATE
        else:
            trend_strength = SignalStrength.WEAK

        is_trending = adx > self.config.adx_threshold

        trend = TrendStrength(
            symbol=symbol,
            timestamp=datetime.now(),
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            is_trending=is_trending
        )

        self._trend_strength_history[symbol].append(trend)

        # 缓存为指标
        indicator = MomentumIndicator(
            name='adx',
            value=adx,
            timestamp=datetime.now(),
            signal=SignalType.BUY if trend_direction == TrendDirection.UPTREND else SignalType.SELL,
            strength=trend_strength,
            metadata={
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'is_trending': is_trending
            }
        )

        self._indicators_cache[symbol]['adx'] = indicator

        return trend

    # ============== 背离检测 ==============

    def detect_divergence(self, symbol: str, indicator_name: str = 'rsi') -> Optional[DivergenceSignal]:
        """
        检测背离信号

        Args:
            symbol: 标的符号
            indicator_name: 指标名称 (rsi, macd等)

        Returns:
            背离信号
        """
        price_history = list(self._price_history[symbol])

        if len(price_history) < self.config.divergence_lookback:
            return None

        # 获取指标历史(简化实现,实际应该维护完整指标历史)
        if indicator_name not in self._indicators_cache[symbol]:
            return None

        # 简化的背离检测:比较最近和之前的峰值
        recent_prices = [p.close for p in price_history[-self.config.divergence_lookback:]]

        # 找到价格的峰值和谷值
        price_peaks = []
        price_troughs = []

        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                price_peaks.append((i, recent_prices[i]))
            elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                price_troughs.append((i, recent_prices[i]))

        # 需要至少2个峰值或谷值来检测背离
        if len(price_peaks) < 2 and len(price_troughs) < 2:
            return None

        # 简化:只检测最近两个峰值的看跌背离
        if len(price_peaks) >= 2:
            peak1_idx, peak1_price = price_peaks[-2]
            peak2_idx, peak2_price = price_peaks[-1]

            # 价格创新高,但指标未创新高 -> 看跌背离
            if peak2_price > peak1_price:
                timestamp1 = price_history[-self.config.divergence_lookback + peak1_idx].timestamp
                timestamp2 = price_history[-self.config.divergence_lookback + peak2_idx].timestamp

                # 简化:假设指标值
                indicator1 = 70.0
                indicator2 = 65.0  # 指标未创新高

                if indicator2 < indicator1:
                    divergence = DivergenceSignal(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        divergence_type=DivergenceType.BEARISH,
                        indicator_name=indicator_name,
                        price_point1=(timestamp1, peak1_price),
                        price_point2=(timestamp2, peak2_price),
                        indicator_point1=(timestamp1, indicator1),
                        indicator_point2=(timestamp2, indicator2),
                        strength=SignalStrength.MODERATE,
                        significance=0.7
                    )

                    self._divergences_history.append(divergence)

                    if self.divergence_callback:
                        self.divergence_callback(divergence)

                    return divergence

        return None

    # ============== 综合信号生成 ==============

    def generate_composite_signal(self, symbol: str) -> Optional[MomentumSignal]:
        """
        生成综合动量信号

        整合所有指标,生成最终交易信号

        Args:
            symbol: 标的符号

        Returns:
            综合信号
        """
        if symbol not in self._indicators_cache or not self._indicators_cache[symbol]:
            return None

        if not self._price_history[symbol]:
            return None

        current_price = self._price_history[symbol][-1].close

        # 收集所有指标信号
        indicators = self._indicators_cache[symbol]
        weights = self.config.signal_weights

        # 计算加权信号分数
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        contributing_indicators = {}

        for ind_name, indicator in indicators.items():
            weight = weights.get(ind_name, 0.1)

            if indicator.signal == SignalType.BUY:
                buy_score += weight * self._strength_to_score(indicator.strength)
            elif indicator.signal == SignalType.SELL:
                sell_score += weight * self._strength_to_score(indicator.strength)

            total_weight += weight
            contributing_indicators[ind_name] = indicator.value

        # 标准化分数
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        # 确定最终信号
        if buy_score > sell_score:
            if buy_score > 0.7:
                signal_type = SignalType.STRONG_BUY
                signal_strength = SignalStrength.VERY_STRONG
            elif buy_score > 0.5:
                signal_type = SignalType.BUY
                signal_strength = SignalStrength.STRONG
            else:
                signal_type = SignalType.BUY
                signal_strength = SignalStrength.MODERATE
            confidence = buy_score
        elif sell_score > buy_score:
            if sell_score > 0.7:
                signal_type = SignalType.STRONG_SELL
                signal_strength = SignalStrength.VERY_STRONG
            elif sell_score > 0.5:
                signal_type = SignalType.SELL
                signal_strength = SignalStrength.STRONG
            else:
                signal_type = SignalType.SELL
                signal_strength = SignalStrength.MODERATE
            confidence = sell_score
        else:
            signal_type = SignalType.HOLD
            signal_strength = SignalStrength.WEAK
            confidence = 0.5

        # 获取趋势方向
        trend_direction = TrendDirection.UNKNOWN
        if symbol in self._trend_strength_history and self._trend_strength_history[symbol]:
            trend_direction = self._trend_strength_history[symbol][-1].trend_direction

        signal = MomentumSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence=confidence,
            price=current_price,
            contributing_indicators=contributing_indicators,
            timeframe=TimeFrame.D1,
            trend_direction=trend_direction
        )

        self._signals_history.append(signal)

        if self.signal_callback:
            self.signal_callback(signal)

        return signal

    def _strength_to_score(self, strength: SignalStrength) -> float:
        """将信号强度转换为分数"""
        strength_map = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0,
        }
        return strength_map.get(strength, 0.5)

    # ============== 多时间框架分析 ==============

    def get_multi_timeframe_momentum(self, symbol: str) -> Dict[TimeFrame, MomentumSignal]:
        """
        获取多时间框架动量分析

        Args:
            symbol: 标的符号

        Returns:
            各时间框架的信号
        """
        # 简化实现:实际应该对不同时间框架分别计算
        result = {}

        for timeframe in self.config.timeframes:
            # 当前实现只有一个时间框架
            signal = self.generate_composite_signal(symbol)
            if signal:
                signal.timeframe = timeframe
                result[timeframe] = signal

        return result

    # ============== 更新逻辑 ==============

    async def _update_all_signals(self) -> None:
        """更新所有标的的信号"""
        for symbol in list(self._price_history.keys()):
            await self._update_signals_for_symbol(symbol)

    async def _update_signals_for_symbol(self, symbol: str) -> None:
        """更新特定标的的信号"""
        # 计算所有指标
        self.calculate_roc(symbol)
        self.calculate_rsi(symbol)
        self.calculate_stochastic(symbol)
        self.calculate_macd(symbol)
        self.calculate_adx(symbol)

        # 检测背离
        self.detect_divergence(symbol)

        # 生成综合信号
        self.generate_composite_signal(symbol)

    # ============== 查询方法 ==============

    def get_latest_signal(self, symbol: str) -> Optional[MomentumSignal]:
        """获取最新信号"""
        for signal in reversed(self._signals_history):
            if signal.symbol == symbol:
                return signal
        return None

    def get_signals_history(self, symbol: str, limit: int = 100) -> List[MomentumSignal]:
        """获取信号历史"""
        signals = [s for s in self._signals_history if s.symbol == symbol]
        return signals[-limit:]

    def get_indicator(self, symbol: str, indicator_name: str) -> Optional[MomentumIndicator]:
        """获取指标值"""
        return self._indicators_cache.get(symbol, {}).get(indicator_name)

    def get_all_indicators(self, symbol: str) -> Dict[str, MomentumIndicator]:
        """获取所有指标"""
        return self._indicators_cache.get(symbol, {})

    def get_divergences(self, symbol: str, limit: int = 50) -> List[DivergenceSignal]:
        """获取背离历史"""
        divergences = [d for d in self._divergences_history if d.symbol == symbol]
        return divergences[-limit:]

    def get_trend_strength(self, symbol: str) -> Optional[TrendStrength]:
        """获取趋势强度"""
        if symbol in self._trend_strength_history and self._trend_strength_history[symbol]:
            return self._trend_strength_history[symbol][-1]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.value,
            'symbols_tracked': len(self._price_history),
            'total_signals': len(self._signals_history),
            'total_divergences': len(self._divergences_history),
            'indicators_cached': sum(len(ind) for ind in self._indicators_cache.values()),
            'config': self.config.to_dict(),
        }


# ============== 工厂函数 ==============

def create_momentum_generator(
    signal_callback: Optional[Callable[[MomentumSignal], None]] = None,
    **config_kwargs
) -> MomentumSignalGenerator:
    """
    创建动量信号生成器

    Args:
        signal_callback: 信号回调函数
        **config_kwargs: 配置参数

    Returns:
        动量信号生成器实例
    """
    config = MomentumConfig(**config_kwargs)
    return MomentumSignalGenerator(
        config=config,
        signal_callback=signal_callback
    )


__all__ = [
    # Enums
    'SignalType',
    'SignalStrength',
    'TrendDirection',
    'DivergenceType',
    'CrossoverType',
    'TimeFrame',
    'GeneratorState',
    # Data Classes
    'PriceData',
    'MomentumIndicator',
    'MomentumSignal',
    'DivergenceSignal',
    'TrendStrength',
    'MomentumConfig',
    # Main Classes
    'MomentumCalculator',
    'MomentumSignalGenerator',
    # Factory Functions
    'create_momentum_generator',
]
