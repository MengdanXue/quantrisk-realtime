"""
Market Regime Detector - 市场状态检测器

实时检测和分类市场状态，包括：
- 趋势/震荡/反转状态识别
- 波动率状态分类 (低/中/高/极端)
- 市场情绪状态 (恐慌/悲观/中性/乐观/狂热)
- 流动性状态监控
- 状态转换检测与预警
- 多时间框架分析
- 状态概率估计

这是量化交易策略调整和风险管理的关键模块。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import statistics
import uuid
import math
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class TrendRegime(str, Enum):
    """趋势状态"""
    STRONG_UPTREND = "strong_uptrend"       # 强上升趋势
    UPTREND = "uptrend"                     # 上升趋势
    SIDEWAYS = "sideways"                   # 震荡横盘
    DOWNTREND = "downtrend"                 # 下降趋势
    STRONG_DOWNTREND = "strong_downtrend"   # 强下降趋势


class VolatilityRegime(str, Enum):
    """波动率状态"""
    VERY_LOW = "very_low"       # 极低波动
    LOW = "low"                 # 低波动
    NORMAL = "normal"           # 正常波动
    HIGH = "high"               # 高波动
    EXTREME = "extreme"         # 极端波动


class SentimentRegime(str, Enum):
    """情绪状态"""
    PANIC = "panic"             # 恐慌
    FEARFUL = "fearful"         # 恐惧
    NEUTRAL = "neutral"         # 中性
    GREEDY = "greedy"           # 贪婪
    EUPHORIC = "euphoric"       # 狂热


class LiquidityRegime(str, Enum):
    """流动性状态"""
    FROZEN = "frozen"           # 流动性枯竭
    THIN = "thin"               # 流动性稀薄
    NORMAL = "normal"           # 正常流动性
    ABUNDANT = "abundant"       # 流动性充裕


class MarketPhase(str, Enum):
    """市场周期阶段"""
    ACCUMULATION = "accumulation"   # 积累阶段
    MARKUP = "markup"               # 上涨阶段
    DISTRIBUTION = "distribution"   # 派发阶段
    MARKDOWN = "markdown"           # 下跌阶段


class RegimeChangeType(str, Enum):
    """状态变化类型"""
    TREND_CHANGE = "trend_change"
    VOLATILITY_CHANGE = "volatility_change"
    SENTIMENT_CHANGE = "sentiment_change"
    LIQUIDITY_CHANGE = "liquidity_change"
    PHASE_CHANGE = "phase_change"


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DetectorState(str, Enum):
    """检测器状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class PriceBar:
    """价格K线"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""

    @property
    def range(self) -> float:
        """K线振幅"""
        return self.high - self.low

    @property
    def body(self) -> float:
        """K线实体"""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """是否阳线"""
        return self.close > self.open

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "range": self.range,
            "body": self.body,
            "is_bullish": self.is_bullish,
        }


@dataclass
class RegimeState:
    """当前状态综合"""
    symbol: str
    timestamp: datetime
    trend: TrendRegime
    volatility: VolatilityRegime
    sentiment: SentimentRegime
    liquidity: LiquidityRegime
    market_phase: MarketPhase
    trend_strength: float = 0.0          # 趋势强度 0-1
    volatility_percentile: float = 0.0   # 波动率百分位 0-100
    sentiment_score: float = 0.0         # 情绪得分 -100 到 100
    liquidity_score: float = 0.0         # 流动性得分 0-100
    confidence: float = 0.0              # 整体置信度 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "sentiment": self.sentiment.value,
            "liquidity": self.liquidity.value,
            "market_phase": self.market_phase.value,
            "trend_strength": self.trend_strength,
            "volatility_percentile": self.volatility_percentile,
            "sentiment_score": self.sentiment_score,
            "liquidity_score": self.liquidity_score,
            "confidence": self.confidence,
        }


@dataclass
class RegimeTransition:
    """状态转换"""
    transition_id: str
    symbol: str
    change_type: RegimeChangeType
    from_state: str
    to_state: str
    timestamp: datetime
    confidence: float
    duration_bars: int = 0              # 持续K线数
    magnitude: float = 0.0              # 变化幅度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "symbol": self.symbol,
            "change_type": self.change_type.value,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "duration_bars": self.duration_bars,
            "magnitude": self.magnitude,
        }


@dataclass
class RegimeProbability:
    """状态概率分布"""
    symbol: str
    timestamp: datetime
    trend_probs: Dict[str, float] = field(default_factory=dict)
    volatility_probs: Dict[str, float] = field(default_factory=dict)
    sentiment_probs: Dict[str, float] = field(default_factory=dict)
    phase_probs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "trend_probs": self.trend_probs,
            "volatility_probs": self.volatility_probs,
            "sentiment_probs": self.sentiment_probs,
            "phase_probs": self.phase_probs,
        }


@dataclass
class RegimeAlert:
    """状态告警"""
    alert_id: str
    symbol: str
    severity: AlertSeverity
    change_type: RegimeChangeType
    message: str
    current_state: str
    previous_state: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "severity": self.severity.value,
            "change_type": self.change_type.value,
            "message": self.message,
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class TrendIndicators:
    """趋势指标"""
    sma_short: float = 0.0
    sma_long: float = 0.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    adx: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    rsi: float = 50.0
    price_position: float = 0.0      # 价格在高低点的位置
    higher_highs: int = 0
    lower_lows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sma_short": self.sma_short,
            "sma_long": self.sma_long,
            "ema_short": self.ema_short,
            "ema_long": self.ema_long,
            "adx": self.adx,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "rsi": self.rsi,
            "price_position": self.price_position,
            "higher_highs": self.higher_highs,
            "lower_lows": self.lower_lows,
        }


@dataclass
class VolatilityIndicators:
    """波动率指标"""
    historical_vol: float = 0.0      # 历史波动率
    atr: float = 0.0                 # ATR
    atr_percent: float = 0.0         # ATR百分比
    bollinger_width: float = 0.0     # 布林带宽度
    vol_percentile: float = 0.0      # 波动率百分位
    vol_of_vol: float = 0.0          # 波动率的波动率
    intraday_range: float = 0.0      # 日内波幅

    def to_dict(self) -> Dict[str, Any]:
        return {
            "historical_vol": self.historical_vol,
            "atr": self.atr,
            "atr_percent": self.atr_percent,
            "bollinger_width": self.bollinger_width,
            "vol_percentile": self.vol_percentile,
            "vol_of_vol": self.vol_of_vol,
            "intraday_range": self.intraday_range,
        }


@dataclass
class DetectorConfig:
    """检测器配置"""
    # 趋势检测参数
    trend_sma_short: int = 20
    trend_sma_long: int = 50
    trend_ema_short: int = 12
    trend_ema_long: int = 26
    adx_period: int = 14
    adx_strong_threshold: float = 25.0

    # 波动率检测参数
    vol_period: int = 20
    atr_period: int = 14
    vol_percentile_lookback: int = 252
    vol_low_threshold: float = 25.0
    vol_high_threshold: float = 75.0
    vol_extreme_threshold: float = 95.0

    # 情绪检测参数
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # 流动性检测参数
    volume_ma_period: int = 20
    volume_thin_threshold: float = 0.5
    volume_abundant_threshold: float = 2.0

    # 监控参数
    update_interval: float = 1.0
    min_bars_for_detection: int = 50
    history_size: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_sma_short": self.trend_sma_short,
            "trend_sma_long": self.trend_sma_long,
            "trend_ema_short": self.trend_ema_short,
            "trend_ema_long": self.trend_ema_long,
            "adx_period": self.adx_period,
            "adx_strong_threshold": self.adx_strong_threshold,
            "vol_period": self.vol_period,
            "atr_period": self.atr_period,
            "vol_percentile_lookback": self.vol_percentile_lookback,
            "vol_low_threshold": self.vol_low_threshold,
            "vol_high_threshold": self.vol_high_threshold,
            "vol_extreme_threshold": self.vol_extreme_threshold,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "volume_ma_period": self.volume_ma_period,
            "volume_thin_threshold": self.volume_thin_threshold,
            "volume_abundant_threshold": self.volume_abundant_threshold,
            "update_interval": self.update_interval,
            "min_bars_for_detection": self.min_bars_for_detection,
            "history_size": self.history_size,
        }


# ============ Providers ============

class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    async def get_price_bars(self, symbol: str, count: int) -> List[PriceBar]:
        """获取价格K线"""
        pass

    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> Optional[PriceBar]:
        """获取最新K线"""
        pass


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self._bars: Dict[str, List[PriceBar]] = {}

    def add_bars(self, symbol: str, bars: List[PriceBar]) -> None:
        """添加K线数据"""
        self._bars[symbol] = bars

    def add_bar(self, symbol: str, bar: PriceBar) -> None:
        """添加单根K线"""
        if symbol not in self._bars:
            self._bars[symbol] = []
        self._bars[symbol].append(bar)

    async def get_price_bars(self, symbol: str, count: int) -> List[PriceBar]:
        """获取价格K线"""
        bars = self._bars.get(symbol, [])
        return bars[-count:] if len(bars) >= count else bars

    async def get_latest_bar(self, symbol: str) -> Optional[PriceBar]:
        """获取最新K线"""
        bars = self._bars.get(symbol, [])
        return bars[-1] if bars else None


# ============ Main Class ============

class MarketRegimeDetector:
    """市场状态检测器"""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        data_provider: Optional[MarketDataProvider] = None,
        regime_callback: Optional[Callable[[RegimeState], None]] = None,
        transition_callback: Optional[Callable[[RegimeTransition], None]] = None,
        alert_callback: Optional[Callable[[RegimeAlert], None]] = None,
    ):
        self.config = config or DetectorConfig()
        self.data_provider = data_provider or simulated_default(SimulatedMarketDataProvider, "data from MarketRegimeDetector's simulated data_provider (SimulatedMarketDataProvider)")
        self.regime_callback = regime_callback
        self.transition_callback = transition_callback
        self.alert_callback = alert_callback

        # 状态
        self._state = DetectorState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._symbols: List[str] = []
        self._current_regimes: Dict[str, RegimeState] = {}
        self._regime_history: Dict[str, deque] = {}
        self._transitions: deque = deque(maxlen=1000)
        self._alerts: deque = deque(maxlen=500)
        self._price_cache: Dict[str, deque] = {}

        # 指标缓存
        self._trend_indicators: Dict[str, TrendIndicators] = {}
        self._volatility_indicators: Dict[str, VolatilityIndicators] = {}

    @property
    def state(self) -> DetectorState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动检测器"""
        if self._state == DetectorState.RUNNING:
            return

        logger.info("Starting Market Regime Detector")
        self._state = DetectorState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Market Regime Detector started")

    async def stop(self) -> None:
        """停止检测器"""
        if self._state == DetectorState.STOPPED:
            return

        logger.info("Stopping Market Regime Detector")
        self._state = DetectorState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Market Regime Detector stopped")

    async def pause(self) -> None:
        """暂停检测器"""
        if self._state == DetectorState.RUNNING:
            self._state = DetectorState.PAUSED
            logger.info("Market Regime Detector paused")

    async def resume(self) -> None:
        """恢复检测器"""
        if self._state == DetectorState.PAUSED:
            self._state = DetectorState.RUNNING
            logger.info("Market Regime Detector resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (DetectorState.RUNNING, DetectorState.PAUSED):
            try:
                if self._state == DetectorState.RUNNING:
                    await self._update_all_regimes()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

    # ============ Symbol Management ============

    def add_symbol(self, symbol: str) -> None:
        """添加监控标的"""
        if symbol not in self._symbols:
            self._symbols.append(symbol)
            self._regime_history[symbol] = deque(maxlen=self.config.history_size)
            self._price_cache[symbol] = deque(maxlen=self.config.history_size)
            logger.info(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """移除监控标的"""
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            self._regime_history.pop(symbol, None)
            self._price_cache.pop(symbol, None)
            self._current_regimes.pop(symbol, None)
            self._trend_indicators.pop(symbol, None)
            self._volatility_indicators.pop(symbol, None)
            logger.info(f"Removed symbol: {symbol}")

    def get_symbols(self) -> List[str]:
        """获取监控标的列表"""
        return list(self._symbols)

    # ============ Price Data ============

    async def add_price_bar(self, bar: PriceBar) -> None:
        """添加价格K线"""
        symbol = bar.symbol
        if symbol not in self._price_cache:
            self._price_cache[symbol] = deque(maxlen=self.config.history_size)

        self._price_cache[symbol].append(bar)

        # 自动添加到监控
        if symbol not in self._symbols:
            self.add_symbol(symbol)

        # 检测状态
        if len(self._price_cache[symbol]) >= self.config.min_bars_for_detection:
            await self._detect_regime(symbol)

    async def _load_price_data(self, symbol: str) -> List[PriceBar]:
        """加载价格数据"""
        # 优先使用缓存
        if symbol in self._price_cache and len(self._price_cache[symbol]) > 0:
            return list(self._price_cache[symbol])

        # 从数据提供者加载
        bars = await self.data_provider.get_price_bars(
            symbol,
            self.config.history_size
        )

        # 更新缓存
        if symbol not in self._price_cache:
            self._price_cache[symbol] = deque(maxlen=self.config.history_size)
        for bar in bars:
            self._price_cache[symbol].append(bar)

        return bars

    # ============ Regime Detection ============

    async def _update_all_regimes(self) -> None:
        """更新所有标的的状态"""
        for symbol in self._symbols:
            await self._detect_regime(symbol)

    async def _detect_regime(self, symbol: str) -> Optional[RegimeState]:
        """检测单个标的的状态"""
        bars = await self._load_price_data(symbol)

        if len(bars) < self.config.min_bars_for_detection:
            return None

        # 计算各类指标
        trend_ind = self._calculate_trend_indicators(bars)
        vol_ind = self._calculate_volatility_indicators(bars)

        self._trend_indicators[symbol] = trend_ind
        self._volatility_indicators[symbol] = vol_ind

        # 判断各维度状态
        trend = self._classify_trend(trend_ind)
        volatility = self._classify_volatility(vol_ind)
        sentiment = self._classify_sentiment(trend_ind, vol_ind)
        liquidity = self._classify_liquidity(bars)
        phase = self._classify_market_phase(trend, trend_ind, vol_ind)

        # 计算各项得分
        trend_strength = self._calculate_trend_strength(trend_ind)
        vol_percentile = vol_ind.vol_percentile
        sentiment_score = self._calculate_sentiment_score(trend_ind)
        liquidity_score = self._calculate_liquidity_score(bars)

        # 计算整体置信度
        confidence = self._calculate_confidence(
            trend_ind, vol_ind, len(bars)
        )

        # 创建状态对象
        new_regime = RegimeState(
            symbol=symbol,
            timestamp=datetime.now(),
            trend=trend,
            volatility=volatility,
            sentiment=sentiment,
            liquidity=liquidity,
            market_phase=phase,
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            sentiment_score=sentiment_score,
            liquidity_score=liquidity_score,
            confidence=confidence,
        )

        # 检测状态变化
        old_regime = self._current_regimes.get(symbol)
        if old_regime:
            self._check_regime_change(old_regime, new_regime)

        # 更新存储
        self._current_regimes[symbol] = new_regime
        self._regime_history[symbol].append(new_regime)

        # 回调
        if self.regime_callback:
            self.regime_callback(new_regime)

        return new_regime

    def _calculate_trend_indicators(self, bars: List[PriceBar]) -> TrendIndicators:
        """计算趋势指标"""
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        # SMA
        sma_short = self._sma(closes, self.config.trend_sma_short)
        sma_long = self._sma(closes, self.config.trend_sma_long)

        # EMA
        ema_short = self._ema(closes, self.config.trend_ema_short)
        ema_long = self._ema(closes, self.config.trend_ema_long)

        # ADX
        adx = self._calculate_adx(highs, lows, closes, self.config.adx_period)

        # MACD
        macd, signal = self._calculate_macd(closes)

        # RSI
        rsi = self._calculate_rsi(closes, self.config.rsi_period)

        # 价格位置
        recent_high = max(highs[-self.config.trend_sma_short:])
        recent_low = min(lows[-self.config.trend_sma_short:])
        price_range = recent_high - recent_low
        price_position = (closes[-1] - recent_low) / price_range if price_range > 0 else 0.5

        # 更高的高点和更低的低点
        higher_highs = self._count_higher_highs(highs, 10)
        lower_lows = self._count_lower_lows(lows, 10)

        return TrendIndicators(
            sma_short=sma_short,
            sma_long=sma_long,
            ema_short=ema_short,
            ema_long=ema_long,
            adx=adx,
            macd=macd,
            macd_signal=signal,
            rsi=rsi,
            price_position=price_position,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
        )

    def _calculate_volatility_indicators(self, bars: List[PriceBar]) -> VolatilityIndicators:
        """计算波动率指标"""
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        # 历史波动率
        returns = self._calculate_returns(closes)
        hist_vol = statistics.stdev(returns) * math.sqrt(252) if len(returns) > 1 else 0

        # ATR
        atr = self._calculate_atr(highs, lows, closes, self.config.atr_period)
        atr_percent = (atr / closes[-1]) * 100 if closes[-1] > 0 else 0

        # 布林带宽度
        sma = self._sma(closes, 20)
        std = statistics.stdev(closes[-20:]) if len(closes) >= 20 else 0
        bollinger_width = (4 * std / sma) * 100 if sma > 0 else 0

        # 波动率百分位
        vol_percentile = self._calculate_vol_percentile(returns)

        # 波动率的波动率
        vol_of_vol = self._calculate_vol_of_vol(returns)

        # 日内波幅
        intraday_ranges = [(b.high - b.low) / b.close * 100 for b in bars[-20:] if b.close > 0]
        intraday_range = statistics.mean(intraday_ranges) if intraday_ranges else 0

        return VolatilityIndicators(
            historical_vol=hist_vol,
            atr=atr,
            atr_percent=atr_percent,
            bollinger_width=bollinger_width,
            vol_percentile=vol_percentile,
            vol_of_vol=vol_of_vol,
            intraday_range=intraday_range,
        )

    def _classify_trend(self, ind: TrendIndicators) -> TrendRegime:
        """分类趋势状态"""
        # 基于多个因素判断
        bullish_signals = 0
        bearish_signals = 0

        # EMA交叉
        if ind.ema_short > ind.ema_long:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # SMA交叉
        if ind.sma_short > ind.sma_long:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # MACD
        if ind.macd > ind.macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # RSI
        if ind.rsi > 50:
            bullish_signals += 1
        elif ind.rsi < 50:
            bearish_signals += 1

        # 价格位置
        if ind.price_position > 0.7:
            bullish_signals += 1
        elif ind.price_position < 0.3:
            bearish_signals += 1

        # 更高高点/更低低点
        if ind.higher_highs > ind.lower_lows:
            bullish_signals += 1
        elif ind.lower_lows > ind.higher_highs:
            bearish_signals += 1

        # 综合判断
        net_signal = bullish_signals - bearish_signals
        is_strong = ind.adx > self.config.adx_strong_threshold

        if net_signal >= 4:
            return TrendRegime.STRONG_UPTREND if is_strong else TrendRegime.UPTREND
        elif net_signal >= 2:
            return TrendRegime.UPTREND
        elif net_signal <= -4:
            return TrendRegime.STRONG_DOWNTREND if is_strong else TrendRegime.DOWNTREND
        elif net_signal <= -2:
            return TrendRegime.DOWNTREND
        else:
            return TrendRegime.SIDEWAYS

    def _classify_volatility(self, ind: VolatilityIndicators) -> VolatilityRegime:
        """分类波动率状态"""
        percentile = ind.vol_percentile

        if percentile >= self.config.vol_extreme_threshold:
            return VolatilityRegime.EXTREME
        elif percentile >= self.config.vol_high_threshold:
            return VolatilityRegime.HIGH
        elif percentile <= 100 - self.config.vol_extreme_threshold:
            return VolatilityRegime.VERY_LOW
        elif percentile <= self.config.vol_low_threshold:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL

    def _classify_sentiment(
        self,
        trend_ind: TrendIndicators,
        vol_ind: VolatilityIndicators
    ) -> SentimentRegime:
        """分类情绪状态"""
        rsi = trend_ind.rsi
        vol = vol_ind.vol_percentile

        # 恐慌: RSI极低 + 高波动
        if rsi < 20 and vol > 80:
            return SentimentRegime.PANIC

        # 狂热: RSI极高 + 高波动
        if rsi > 80 and vol > 70:
            return SentimentRegime.EUPHORIC

        # 恐惧: RSI较低
        if rsi < self.config.rsi_oversold:
            return SentimentRegime.FEARFUL

        # 贪婪: RSI较高
        if rsi > self.config.rsi_overbought:
            return SentimentRegime.GREEDY

        return SentimentRegime.NEUTRAL

    def _classify_liquidity(self, bars: List[PriceBar]) -> LiquidityRegime:
        """分类流动性状态"""
        if not bars:
            return LiquidityRegime.NORMAL

        volumes = [b.volume for b in bars]
        if not volumes:
            return LiquidityRegime.NORMAL

        avg_volume = statistics.mean(volumes[-self.config.volume_ma_period:])
        recent_volume = volumes[-1]

        if avg_volume == 0:
            return LiquidityRegime.FROZEN

        volume_ratio = recent_volume / avg_volume

        if volume_ratio < 0.2:
            return LiquidityRegime.FROZEN
        elif volume_ratio < self.config.volume_thin_threshold:
            return LiquidityRegime.THIN
        elif volume_ratio > self.config.volume_abundant_threshold:
            return LiquidityRegime.ABUNDANT
        else:
            return LiquidityRegime.NORMAL

    def _classify_market_phase(
        self,
        trend: TrendRegime,
        trend_ind: TrendIndicators,
        vol_ind: VolatilityIndicators,
    ) -> MarketPhase:
        """分类市场周期阶段"""
        # 简化的Wyckoff周期判断

        # 上涨阶段: 上升趋势 + 成交量确认
        if trend in (TrendRegime.UPTREND, TrendRegime.STRONG_UPTREND):
            if trend_ind.price_position > 0.5:
                return MarketPhase.MARKUP
            else:
                return MarketPhase.ACCUMULATION

        # 下跌阶段: 下降趋势
        if trend in (TrendRegime.DOWNTREND, TrendRegime.STRONG_DOWNTREND):
            if trend_ind.price_position < 0.5:
                return MarketPhase.MARKDOWN
            else:
                return MarketPhase.DISTRIBUTION

        # 震荡: 可能是积累或派发
        if trend_ind.rsi > 50:
            return MarketPhase.DISTRIBUTION
        else:
            return MarketPhase.ACCUMULATION

    # ============ Score Calculations ============

    def _calculate_trend_strength(self, ind: TrendIndicators) -> float:
        """计算趋势强度 0-1"""
        # 综合ADX、EMA差距、价格位置
        adx_score = min(ind.adx / 50, 1.0)  # ADX标准化

        ema_diff = abs(ind.ema_short - ind.ema_long)
        ema_score = min(ema_diff / ind.ema_long * 10, 1.0) if ind.ema_long > 0 else 0

        position_score = abs(ind.price_position - 0.5) * 2  # 距离中心的距离

        return (adx_score * 0.4 + ema_score * 0.3 + position_score * 0.3)

    def _calculate_sentiment_score(self, ind: TrendIndicators) -> float:
        """计算情绪得分 -100 到 100"""
        # RSI映射到 -100 到 100
        return (ind.rsi - 50) * 2

    def _calculate_liquidity_score(self, bars: List[PriceBar]) -> float:
        """计算流动性得分 0-100"""
        if not bars:
            return 50.0

        volumes = [b.volume for b in bars]
        if len(volumes) < self.config.volume_ma_period:
            return 50.0

        avg_volume = statistics.mean(volumes[-self.config.volume_ma_period:])
        recent_volume = volumes[-1]

        if avg_volume == 0:
            return 0.0

        ratio = recent_volume / avg_volume
        # 标准化到 0-100
        return min(ratio * 50, 100.0)

    def _calculate_confidence(
        self,
        trend_ind: TrendIndicators,
        vol_ind: VolatilityIndicators,
        bar_count: int
    ) -> float:
        """计算整体置信度 0-1"""
        # 数据充分性
        data_score = min(bar_count / 100, 1.0)

        # ADX信号强度
        adx_score = min(trend_ind.adx / 30, 1.0)

        # 波动率稳定性 (波动率的波动率越低越稳定)
        vol_stability = max(0, 1 - vol_ind.vol_of_vol / 0.5)

        return (data_score * 0.3 + adx_score * 0.4 + vol_stability * 0.3)

    # ============ Regime Change Detection ============

    def _check_regime_change(self, old: RegimeState, new: RegimeState) -> None:
        """检测状态变化"""
        # 趋势变化
        if old.trend != new.trend:
            self._create_transition(
                new.symbol,
                RegimeChangeType.TREND_CHANGE,
                old.trend.value,
                new.trend.value,
                new.confidence,
            )

        # 波动率变化
        if old.volatility != new.volatility:
            self._create_transition(
                new.symbol,
                RegimeChangeType.VOLATILITY_CHANGE,
                old.volatility.value,
                new.volatility.value,
                new.confidence,
            )

        # 情绪变化
        if old.sentiment != new.sentiment:
            self._create_transition(
                new.symbol,
                RegimeChangeType.SENTIMENT_CHANGE,
                old.sentiment.value,
                new.sentiment.value,
                new.confidence,
            )

        # 流动性变化
        if old.liquidity != new.liquidity:
            self._create_transition(
                new.symbol,
                RegimeChangeType.LIQUIDITY_CHANGE,
                old.liquidity.value,
                new.liquidity.value,
                new.confidence,
            )

        # 周期阶段变化
        if old.market_phase != new.market_phase:
            self._create_transition(
                new.symbol,
                RegimeChangeType.PHASE_CHANGE,
                old.market_phase.value,
                new.market_phase.value,
                new.confidence,
            )

    def _create_transition(
        self,
        symbol: str,
        change_type: RegimeChangeType,
        from_state: str,
        to_state: str,
        confidence: float,
    ) -> None:
        """创建状态转换记录"""
        transition = RegimeTransition(
            transition_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            change_type=change_type,
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            confidence=confidence,
        )

        self._transitions.append(transition)

        # 回调
        if self.transition_callback:
            self.transition_callback(transition)

        # 根据变化类型创建告警
        severity = self._get_alert_severity(change_type, from_state, to_state)
        self._create_alert(symbol, change_type, to_state, from_state, severity)

    def _get_alert_severity(
        self,
        change_type: RegimeChangeType,
        from_state: str,
        to_state: str
    ) -> AlertSeverity:
        """确定告警严重程度"""
        # 关键变化
        critical_changes = [
            ("uptrend", "downtrend"),
            ("downtrend", "uptrend"),
            ("normal", "extreme"),
            ("neutral", "panic"),
            ("normal", "frozen"),
        ]

        warning_changes = [
            ("sideways", "downtrend"),
            ("sideways", "uptrend"),
            ("low", "high"),
            ("neutral", "fearful"),
            ("normal", "thin"),
        ]

        change = (from_state, to_state)
        reverse_change = (to_state, from_state)

        if change in critical_changes or reverse_change in critical_changes:
            return AlertSeverity.CRITICAL
        elif change in warning_changes or reverse_change in warning_changes:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _create_alert(
        self,
        symbol: str,
        change_type: RegimeChangeType,
        current_state: str,
        previous_state: str,
        severity: AlertSeverity,
    ) -> None:
        """创建告警"""
        alert = RegimeAlert(
            alert_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            severity=severity,
            change_type=change_type,
            message=f"{symbol} {change_type.value}: {previous_state} -> {current_state}",
            current_state=current_state,
            previous_state=previous_state,
        )

        self._alerts.append(alert)

        if self.alert_callback:
            self.alert_callback(alert)

    # ============ Technical Indicator Calculations ============

    def _sma(self, data: List[float], period: int) -> float:
        """计算简单移动平均"""
        if len(data) < period:
            return data[-1] if data else 0
        return statistics.mean(data[-period:])

    def _ema(self, data: List[float], period: int) -> float:
        """计算指数移动平均"""
        if len(data) < period:
            return data[-1] if data else 0

        multiplier = 2 / (period + 1)
        ema = data[0]

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_returns(self, closes: List[float]) -> List[float]:
        """计算收益率序列"""
        if len(closes) < 2:
            return []
        return [(closes[i] - closes[i-1]) / closes[i-1]
                for i in range(1, len(closes)) if closes[i-1] != 0]

    def _calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int
    ) -> float:
        """计算ATR"""
        if len(closes) < 2:
            return 0

        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)

        if len(trs) < period:
            return statistics.mean(trs) if trs else 0

        return statistics.mean(trs[-period:])

    def _calculate_adx(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int
    ) -> float:
        """计算ADX (简化版)"""
        if len(closes) < period + 1:
            return 0

        # 计算方向移动
        plus_dm = []
        minus_dm = []

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        atr = self._calculate_atr(highs, lows, closes, period)
        if atr == 0:
            return 0

        # 平滑DI
        plus_di = (statistics.mean(plus_dm[-period:]) / atr) * 100 if plus_dm else 0
        minus_di = (statistics.mean(minus_dm[-period:]) / atr) * 100 if minus_dm else 0

        # 计算DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0

        dx = abs(plus_di - minus_di) / di_sum * 100

        return dx

    def _calculate_macd(
        self,
        closes: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float]:
        """计算MACD"""
        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)
        macd = ema_fast - ema_slow

        # MACD信号线需要MACD历史
        macd_history = []
        for i in range(slow, len(closes)):
            ef = self._ema(closes[:i+1], fast)
            es = self._ema(closes[:i+1], slow)
            macd_history.append(ef - es)

        signal_line = self._ema(macd_history, signal) if len(macd_history) >= signal else macd

        return macd, signal_line

    def _calculate_rsi(self, closes: List[float], period: int) -> float:
        """计算RSI"""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = statistics.mean(gains[-period:]) if gains else 0
        avg_loss = statistics.mean(losses[-period:]) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_vol_percentile(self, returns: List[float]) -> float:
        """计算波动率百分位"""
        if len(returns) < 20:
            return 50.0

        # 滚动波动率
        window = 20
        vols = []
        for i in range(window, len(returns)):
            vol = statistics.stdev(returns[i-window:i])
            vols.append(vol)

        if not vols:
            return 50.0

        current_vol = vols[-1]
        sorted_vols = sorted(vols)

        # 计算百分位
        rank = sum(1 for v in sorted_vols if v < current_vol)
        percentile = (rank / len(sorted_vols)) * 100

        return percentile

    def _calculate_vol_of_vol(self, returns: List[float]) -> float:
        """计算波动率的波动率"""
        if len(returns) < 40:
            return 0

        window = 20
        vols = []
        for i in range(window, len(returns)):
            vol = statistics.stdev(returns[i-window:i])
            vols.append(vol)

        if len(vols) < 2:
            return 0

        return statistics.stdev(vols)

    def _count_higher_highs(self, highs: List[float], lookback: int) -> int:
        """统计更高的高点数量"""
        if len(highs) < lookback + 1:
            return 0

        count = 0
        recent = highs[-lookback:]
        for i in range(1, len(recent)):
            if recent[i] > recent[i-1]:
                count += 1
        return count

    def _count_lower_lows(self, lows: List[float], lookback: int) -> int:
        """统计更低的低点数量"""
        if len(lows) < lookback + 1:
            return 0

        count = 0
        recent = lows[-lookback:]
        for i in range(1, len(recent)):
            if recent[i] < recent[i-1]:
                count += 1
        return count

    # ============ Query Methods ============

    def get_current_regime(self, symbol: str) -> Optional[RegimeState]:
        """获取当前状态"""
        return self._current_regimes.get(symbol)

    def get_all_regimes(self) -> Dict[str, RegimeState]:
        """获取所有标的的当前状态"""
        return dict(self._current_regimes)

    def get_regime_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[RegimeState]:
        """获取状态历史"""
        history = self._regime_history.get(symbol, deque())
        return list(history)[-limit:]

    def get_transitions(
        self,
        symbol: Optional[str] = None,
        change_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[RegimeTransition]:
        """获取状态转换"""
        transitions = list(self._transitions)

        if symbol:
            transitions = [t for t in transitions if t.symbol == symbol]
        if change_type:
            transitions = [t for t in transitions if t.change_type.value == change_type]

        return transitions[-limit:]

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[RegimeAlert]:
        """获取告警"""
        alerts = list(self._alerts)

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_regime_probability(self, symbol: str) -> Optional[RegimeProbability]:
        """获取状态概率分布"""
        history = self._regime_history.get(symbol, deque())
        if len(history) < 10:
            return None

        # 统计各状态出现频率
        trend_counts: Dict[str, int] = {}
        vol_counts: Dict[str, int] = {}
        sentiment_counts: Dict[str, int] = {}
        phase_counts: Dict[str, int] = {}

        for regime in history:
            trend_counts[regime.trend.value] = trend_counts.get(regime.trend.value, 0) + 1
            vol_counts[regime.volatility.value] = vol_counts.get(regime.volatility.value, 0) + 1
            sentiment_counts[regime.sentiment.value] = sentiment_counts.get(regime.sentiment.value, 0) + 1
            phase_counts[regime.market_phase.value] = phase_counts.get(regime.market_phase.value, 0) + 1

        total = len(history)

        return RegimeProbability(
            symbol=symbol,
            timestamp=datetime.now(),
            trend_probs={k: v/total for k, v in trend_counts.items()},
            volatility_probs={k: v/total for k, v in vol_counts.items()},
            sentiment_probs={k: v/total for k, v in sentiment_counts.items()},
            phase_probs={k: v/total for k, v in phase_counts.items()},
        )

    def get_trend_indicators(self, symbol: str) -> Optional[TrendIndicators]:
        """获取趋势指标"""
        return self._trend_indicators.get(symbol)

    def get_volatility_indicators(self, symbol: str) -> Optional[VolatilityIndicators]:
        """获取波动率指标"""
        return self._volatility_indicators.get(symbol)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "symbols": len(self._symbols),
            "current_regimes": len(self._current_regimes),
            "total_transitions": len(self._transitions),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "config": self.config.to_dict(),
        }


# ============ Factory Functions ============

def create_regime_detector(
    data_provider: Optional[MarketDataProvider] = None,
    **config_kwargs,
) -> MarketRegimeDetector:
    """创建市场状态检测器"""
    config = DetectorConfig(**config_kwargs)
    return MarketRegimeDetector(
        config=config,
        data_provider=data_provider,
    )


def create_price_bar(
    timestamp: datetime,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    symbol: str = "",
) -> PriceBar:
    """创建价格K线"""
    return PriceBar(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
    )


async def quick_regime_check(
    detector: MarketRegimeDetector,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """快速状态检查"""
    regime = await detector._detect_regime(symbol)
    if regime:
        return regime.to_dict()
    return None


# ============ Exports ============

__all__ = [
    # Enums
    "TrendRegime",
    "VolatilityRegime",
    "SentimentRegime",
    "LiquidityRegime",
    "MarketPhase",
    "RegimeChangeType",
    "AlertSeverity",
    "DetectorState",
    # Data Classes
    "PriceBar",
    "RegimeState",
    "RegimeTransition",
    "RegimeProbability",
    "RegimeAlert",
    "TrendIndicators",
    "VolatilityIndicators",
    "DetectorConfig",
    # Providers
    "MarketDataProvider",
    "SimulatedMarketDataProvider",
    # Main Class
    "MarketRegimeDetector",
    # Factory Functions
    "create_regime_detector",
    "create_price_bar",
    "quick_regime_check",
]
