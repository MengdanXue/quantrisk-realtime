"""
Derivatives Indicators - 衍生品指标模块

实时衍生品市场指标计算和监控系统。
支持期权流向分析、波动率偏度、期限结构等高级指标。

核心功能：
1. PCR指标 - Put-Call Ratio（看跌看涨比率）
2. 期权异常活动检测 - 大单、异常成交量、期权流向
3. 波动率偏度指标 - Skew Index, Risk Reversal, Butterfly
4. 期权情绪指标 - Smart Money Index, Option Pain, Max Pain
5. 期限结构指标 - Contango/Backwardation, Roll Yield, Term Premium
6. 实时监控和警报 - 异常检测和智能告警
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Constants ============


# 期权异常检测阈值
BLOCK_TRADE_MIN_SIZE = 100  # 大单最小规模
UNUSUAL_VOLUME_MULTIPLIER = 3.0  # 异常成交量倍数


# ============ Enums ============


class OptionType(str, Enum):
    """期权类型"""
    CALL = "call"
    PUT = "put"


class PCRType(str, Enum):
    """PCR类型"""
    VOLUME = "volume"           # 成交量PCR
    OPEN_INTEREST = "oi"        # 持仓量PCR
    WEIGHTED = "weighted"       # 加权PCR
    DELTA_ADJUSTED = "delta"    # Delta调整PCR


class FlowDirection(str, Enum):
    """期权流向"""
    BULLISH = "bullish"         # 看涨
    BEARISH = "bearish"         # 看跌
    NEUTRAL = "neutral"         # 中性
    MIXED = "mixed"             # 混合


class ActivityType(str, Enum):
    """活动类型"""
    BLOCK_TRADE = "block"       # 大单交易
    SWEEP = "sweep"             # 扫单
    SPLIT = "split"             # 拆单
    UNUSUAL_VOLUME = "volume"   # 异常成交量
    OI_BUILDUP = "oi_buildup"   # 持仓增加


class TermStructureShape(str, Enum):
    """期限结构形态"""
    CONTANGO = "contango"               # 正向市场
    BACKWARDATION = "backwardation"     # 反向市场
    FLAT = "flat"                       # 平坦
    INVERTED = "inverted"               # 倒挂
    HUMPED = "humped"                   # 驼峰


class SentimentLevel(str, Enum):
    """情绪水平"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class AlertSeverity(str, Enum):
    """警报严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """警报类型"""
    PCR_EXTREME = "pcr_extreme"
    UNUSUAL_ACTIVITY = "unusual_activity"
    SKEW_ANOMALY = "skew_anomaly"
    SENTIMENT_SHIFT = "sentiment_shift"
    TERM_STRUCTURE_CHANGE = "term_structure"
    OPTION_PAIN = "option_pain"


class MonitorState(str, Enum):
    """监控器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============


@dataclass
class OptionData:
    """期权数据"""
    symbol: str                     # 标的代码
    option_symbol: str              # 期权代码
    option_type: OptionType         # 期权类型
    strike: float                   # 行权价
    expiry: datetime                # 到期日
    spot_price: float               # 标的价格
    volume: int                     # 成交量
    open_interest: int              # 持仓量
    bid: float                      # 买价
    ask: float                      # 卖价
    last: float                     # 最新价
    implied_vol: Optional[float] = None     # 隐含波动率
    delta: Optional[float] = None           # Delta
    gamma: Optional[float] = None           # Gamma
    theta: Optional[float] = None           # Theta
    vega: Optional[float] = None            # Vega
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "spot_price": self.spot_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "implied_vol": self.implied_vol,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PCRIndicator:
    """PCR指标"""
    symbol: str
    timestamp: datetime
    pcr_type: PCRType
    pcr_value: float                # PCR值
    put_volume: int                 # 看跌成交量
    call_volume: int                # 看涨成交量
    put_oi: int                     # 看跌持仓量
    call_oi: int                    # 看涨持仓量
    interpretation: SentimentLevel  # 解读
    percentile_1m: Optional[float] = None  # 1月百分位
    percentile_3m: Optional[float] = None  # 3月百分位
    percentile_1y: Optional[float] = None  # 1年百分位

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "pcr_type": self.pcr_type.value,
            "pcr_value": self.pcr_value,
            "put_volume": self.put_volume,
            "call_volume": self.call_volume,
            "put_oi": self.put_oi,
            "call_oi": self.call_oi,
            "interpretation": self.interpretation.value,
            "percentile_1m": self.percentile_1m,
            "percentile_3m": self.percentile_3m,
            "percentile_1y": self.percentile_1y,
        }


@dataclass
class UnusualActivity:
    """异常活动"""
    symbol: str
    option_symbol: str
    activity_type: ActivityType
    option_type: OptionType
    strike: float
    expiry: datetime
    volume: int
    avg_volume: float               # 平均成交量
    volume_ratio: float             # 成交量比率
    oi_change: int                  # 持仓量变化
    premium: float                  # 权利金
    notional_value: float           # 名义价值
    flow_direction: FlowDirection   # 流向
    confidence: float               # 置信度
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "activity_type": self.activity_type.value,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "volume_ratio": self.volume_ratio,
            "oi_change": self.oi_change,
            "premium": self.premium,
            "notional_value": self.notional_value,
            "flow_direction": self.flow_direction.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VolatilitySkew:
    """波动率偏度"""
    symbol: str
    expiry: datetime
    timestamp: datetime
    atm_strike: float               # ATM行权价
    atm_vol: float                  # ATM波动率
    skew_index: float               # 偏度指数
    risk_reversal_25d: float        # 25D风险逆转
    risk_reversal_10d: float        # 10D风险逆转
    butterfly_25d: float            # 25D蝶式
    butterfly_10d: float            # 10D蝶式
    put_skew: float                 # 看跌偏度
    call_skew: float                # 看涨偏度
    skew_slope: float               # 偏度斜率
    smile_convexity: float          # 微笑凸度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "atm_strike": self.atm_strike,
            "atm_vol": self.atm_vol,
            "skew_index": self.skew_index,
            "risk_reversal_25d": self.risk_reversal_25d,
            "risk_reversal_10d": self.risk_reversal_10d,
            "butterfly_25d": self.butterfly_25d,
            "butterfly_10d": self.butterfly_10d,
            "put_skew": self.put_skew,
            "call_skew": self.call_skew,
            "skew_slope": self.skew_slope,
            "smile_convexity": self.smile_convexity,
        }


@dataclass
class OptionSentiment:
    """期权情绪指标"""
    symbol: str
    timestamp: datetime
    smart_money_index: float        # 聪明钱指数
    option_pain: float              # 期权痛点
    max_pain_strike: float          # 最大痛点行权价
    pain_distance: float            # 痛点距离（%）
    call_put_ratio: float           # 看涨看跌比率
    put_call_parity: float          # 看跌看涨平价
    sentiment_score: float          # 情绪分数（-100到100）
    sentiment_level: SentimentLevel # 情绪水平
    bullish_flow: float             # 看涨流入
    bearish_flow: float             # 看跌流入

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "smart_money_index": self.smart_money_index,
            "option_pain": self.option_pain,
            "max_pain_strike": self.max_pain_strike,
            "pain_distance": self.pain_distance,
            "call_put_ratio": self.call_put_ratio,
            "put_call_parity": self.put_call_parity,
            "sentiment_score": self.sentiment_score,
            "sentiment_level": self.sentiment_level.value,
            "bullish_flow": self.bullish_flow,
            "bearish_flow": self.bearish_flow,
        }


@dataclass
class FuturesContract:
    """期货合约"""
    symbol: str
    contract_month: str             # 合约月份
    expiry: datetime                # 到期日
    price: float                    # 价格
    volume: int                     # 成交量
    open_interest: int              # 持仓量
    basis: float                    # 基差
    time_to_expiry: float           # 到期时间（年）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "contract_month": self.contract_month,
            "expiry": self.expiry.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "basis": self.basis,
            "time_to_expiry": self.time_to_expiry,
        }


@dataclass
class TermStructure:
    """期限结构"""
    symbol: str
    timestamp: datetime
    spot_price: float               # 现货价格
    contracts: List[FuturesContract] # 期货合约
    shape: TermStructureShape       # 形态
    front_month_price: float        # 近月价格
    back_month_price: float         # 远月价格
    contango_pct: float             # 升水百分比
    roll_yield: float               # 展期收益
    term_premium: float             # 期限溢价
    convenience_yield: float        # 便利收益率
    carrying_cost: float            # 持有成本

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "spot_price": self.spot_price,
            "contracts": [c.to_dict() for c in self.contracts],
            "shape": self.shape.value,
            "front_month_price": self.front_month_price,
            "back_month_price": self.back_month_price,
            "contango_pct": self.contango_pct,
            "roll_yield": self.roll_yield,
            "term_premium": self.term_premium,
            "convenience_yield": self.convenience_yield,
            "carrying_cost": self.carrying_cost,
        }


@dataclass
class DerivativesAlert:
    """衍生品警报"""
    alert_id: str
    symbol: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DerivativesSnapshot:
    """衍生品快照"""
    symbol: str
    timestamp: datetime
    pcr_indicators: Dict[PCRType, PCRIndicator]
    unusual_activities: List[UnusualActivity]
    volatility_skews: List[VolatilitySkew]
    sentiment: OptionSentiment
    term_structure: Optional[TermStructure]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "pcr_indicators": {k.value: v.to_dict() for k, v in self.pcr_indicators.items()},
            "unusual_activities": [a.to_dict() for a in self.unusual_activities],
            "volatility_skews": [s.to_dict() for s in self.volatility_skews],
            "sentiment": self.sentiment.to_dict(),
            "term_structure": self.term_structure.to_dict() if self.term_structure else None,
        }


@dataclass
class DerivativesConfig:
    """衍生品指标配置"""
    update_interval_ms: int = 5000              # 更新间隔
    history_size: int = 1000                    # 历史记录大小
    # PCR配置
    pcr_extreme_low: float = 0.5                # PCR极低阈值
    pcr_extreme_high: float = 1.5               # PCR极高阈值
    # 异常活动配置
    block_trade_min_contracts: int = 100        # 大单最小合约数
    unusual_volume_threshold: float = 3.0       # 异常成交量阈值
    oi_buildup_threshold: float = 0.2           # 持仓增加阈值
    # 偏度配置
    skew_anomaly_threshold: float = 0.1         # 偏度异常阈值
    # 情绪配置
    sentiment_extreme_threshold: float = 80     # 极端情绪阈值
    # 期限结构配置
    contango_threshold: float = 0.05            # 升水阈值
    backwardation_threshold: float = -0.05      # 贴水阈值


# ============ Abstract Interfaces ============


class OptionDataProvider(ABC):
    """期权数据提供者接口"""

    @abstractmethod
    async def get_option_chain(self, symbol: str) -> List[OptionData]:
        """获取期权链"""
        pass

    @abstractmethod
    async def get_option_history(
        self, symbol: str, option_symbol: str, days: int
    ) -> List[OptionData]:
        """获取期权历史数据"""
        pass


class FuturesDataProvider(ABC):
    """期货数据提供者接口"""

    @abstractmethod
    async def get_futures_chain(self, symbol: str) -> List[FuturesContract]:
        """获取期货链"""
        pass

    @abstractmethod
    async def get_spot_price(self, symbol: str) -> float:
        """获取现货价格"""
        pass


# ============ Simulated Providers ============


class SimulatedOptionDataProvider(OptionDataProvider):
    """模拟期权数据提供者"""

    def __init__(self):
        self._option_chains: Dict[str, List[OptionData]] = {}
        self._option_history: Dict[str, List[OptionData]] = {}

    def set_option_chain(self, symbol: str, chain: List[OptionData]) -> None:
        """设置期权链"""
        self._option_chains[symbol] = chain

    def set_option_history(
        self, symbol: str, option_symbol: str, history: List[OptionData]
    ) -> None:
        """设置期权历史"""
        key = f"{symbol}:{option_symbol}"
        self._option_history[key] = history

    async def get_option_chain(self, symbol: str) -> List[OptionData]:
        return self._option_chains.get(symbol, [])

    async def get_option_history(
        self, symbol: str, option_symbol: str, days: int
    ) -> List[OptionData]:
        key = f"{symbol}:{option_symbol}"
        history = self._option_history.get(key, [])
        return history[-days:] if history else []


class SimulatedFuturesDataProvider(FuturesDataProvider):
    """模拟期货数据提供者"""

    def __init__(self):
        self._futures_chains: Dict[str, List[FuturesContract]] = {}
        self._spot_prices: Dict[str, float] = {}

    def set_futures_chain(self, symbol: str, chain: List[FuturesContract]) -> None:
        """设置期货链"""
        self._futures_chains[symbol] = chain

    def set_spot_price(self, symbol: str, price: float) -> None:
        """设置现货价格"""
        self._spot_prices[symbol] = price

    async def get_futures_chain(self, symbol: str) -> List[FuturesContract]:
        return self._futures_chains.get(symbol, [])

    async def get_spot_price(self, symbol: str) -> float:
        return self._spot_prices.get(symbol, 100.0)


# ============ Main Class ============


class DerivativesIndicators:
    """衍生品指标计算器"""

    def __init__(
        self,
        config: DerivativesConfig,
        option_provider: Optional[OptionDataProvider] = None,
        futures_provider: Optional[FuturesDataProvider] = None,
    ):
        self._config = config
        self._option_provider = option_provider or simulated_default(SimulatedOptionDataProvider, "data from DerivativesIndicators's simulated option_provider (SimulatedOptionDataProvider)")
        self._futures_provider = futures_provider or simulated_default(SimulatedFuturesDataProvider, "data from DerivativesIndicators's simulated futures_provider (SimulatedFuturesDataProvider)")

        # 状态
        self._state = MonitorState.IDLE
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 符号追踪
        self._symbols: Set[str] = set()

        # 缓存
        self._pcr_cache: Dict[str, Dict[PCRType, PCRIndicator]] = {}
        self._activities_cache: Dict[str, List[UnusualActivity]] = {}
        self._skew_cache: Dict[str, List[VolatilitySkew]] = {}
        self._sentiment_cache: Dict[str, OptionSentiment] = {}
        self._term_structure_cache: Dict[str, TermStructure] = {}

        # 历史记录
        self._snapshot_history: Dict[str, List[DerivativesSnapshot]] = {}
        self._alert_history: List[DerivativesAlert] = []
        self._alert_counter = 0

        # 回调
        self._snapshot_callbacks: List[Callable[[DerivativesSnapshot], None]] = []
        self._alert_callbacks: List[Callable[[DerivativesAlert], None]] = []

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动监控器"""
        if self._running:
            return

        self._running = True
        self._state = MonitorState.RUNNING
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("DerivativesIndicators started")

    async def stop(self) -> None:
        """停止监控器"""
        self._running = False
        self._state = MonitorState.IDLE
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("DerivativesIndicators stopped")

    async def _update_loop(self) -> None:
        """更新循环"""
        while self._running:
            try:
                for symbol in self._symbols:
                    await self._update_indicators(symbol)

                await asyncio.sleep(self._config.update_interval_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1)

    # ============ Symbol Management ============

    def add_symbol(self, symbol: str) -> None:
        """添加监控标的"""
        self._symbols.add(symbol)
        self._snapshot_history[symbol] = []

    def remove_symbol(self, symbol: str) -> None:
        """移除监控标的"""
        self._symbols.discard(symbol)
        self._pcr_cache.pop(symbol, None)
        self._activities_cache.pop(symbol, None)
        self._skew_cache.pop(symbol, None)
        self._sentiment_cache.pop(symbol, None)
        self._term_structure_cache.pop(symbol, None)

    def get_symbols(self) -> List[str]:
        """获取监控标的列表"""
        return list(self._symbols)

    # ============ Indicator Calculation ============

    async def _update_indicators(self, symbol: str) -> None:
        """更新指标"""
        try:
            # 获取期权链数据
            option_chain = await self._option_provider.get_option_chain(symbol)
            if not option_chain:
                return

            # 计算各类指标
            pcr_indicators = await self._calculate_pcr(symbol, option_chain)
            unusual_activities = await self._detect_unusual_activity(symbol, option_chain)
            volatility_skews = await self._calculate_volatility_skew(symbol, option_chain)
            sentiment = await self._calculate_sentiment(symbol, option_chain)

            # 获取期限结构（如果有期货数据）
            term_structure = await self._calculate_term_structure(symbol)

            # 更新缓存
            self._pcr_cache[symbol] = pcr_indicators
            self._activities_cache[symbol] = unusual_activities
            self._skew_cache[symbol] = volatility_skews
            self._sentiment_cache[symbol] = sentiment
            if term_structure:
                self._term_structure_cache[symbol] = term_structure

            # 创建快照
            snapshot = DerivativesSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                pcr_indicators=pcr_indicators,
                unusual_activities=unusual_activities,
                volatility_skews=volatility_skews,
                sentiment=sentiment,
                term_structure=term_structure,
            )

            # 记录历史
            self._add_to_history(symbol, snapshot)

            # 检查警报
            alerts = self._check_alerts(symbol, snapshot)
            for alert in alerts:
                self._alert_history.append(alert)
                self._notify_alert(alert)

            # 通知回调
            self._notify_snapshot(snapshot)

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}")

    # ============ PCR Calculation ============

    async def _calculate_pcr(
        self, symbol: str, option_chain: List[OptionData]
    ) -> Dict[PCRType, PCRIndicator]:
        """计算PCR指标"""
        now = datetime.now()
        indicators = {}

        # 分离看涨和看跌
        calls = [opt for opt in option_chain if opt.option_type == OptionType.CALL]
        puts = [opt for opt in option_chain if opt.option_type == OptionType.PUT]

        # 成交量统计
        call_volume = sum(c.volume for c in calls)
        put_volume = sum(p.volume for p in puts)
        call_oi = sum(c.open_interest for c in calls)
        put_oi = sum(p.open_interest for p in puts)

        # 计算Volume PCR
        if call_volume > 0:
            volume_pcr = put_volume / call_volume
            indicators[PCRType.VOLUME] = PCRIndicator(
                symbol=symbol,
                timestamp=now,
                pcr_type=PCRType.VOLUME,
                pcr_value=volume_pcr,
                put_volume=put_volume,
                call_volume=call_volume,
                put_oi=put_oi,
                call_oi=call_oi,
                interpretation=self._interpret_pcr(volume_pcr),
            )

        # 计算Open Interest PCR
        if call_oi > 0:
            oi_pcr = put_oi / call_oi
            indicators[PCRType.OPEN_INTEREST] = PCRIndicator(
                symbol=symbol,
                timestamp=now,
                pcr_type=PCRType.OPEN_INTEREST,
                pcr_value=oi_pcr,
                put_volume=put_volume,
                call_volume=call_volume,
                put_oi=put_oi,
                call_oi=call_oi,
                interpretation=self._interpret_pcr(oi_pcr),
            )

        # 计算加权PCR（成交量和持仓量的加权平均）
        if call_volume > 0 and call_oi > 0:
            weighted_pcr = 0.6 * (put_volume / call_volume) + 0.4 * (put_oi / call_oi)
            indicators[PCRType.WEIGHTED] = PCRIndicator(
                symbol=symbol,
                timestamp=now,
                pcr_type=PCRType.WEIGHTED,
                pcr_value=weighted_pcr,
                put_volume=put_volume,
                call_volume=call_volume,
                put_oi=put_oi,
                call_oi=call_oi,
                interpretation=self._interpret_pcr(weighted_pcr),
            )

        # 计算Delta调整PCR
        call_delta_sum = sum(abs(c.delta) * c.open_interest for c in calls if c.delta)
        put_delta_sum = sum(abs(p.delta) * p.open_interest for p in puts if p.delta)
        if call_delta_sum > 0:
            delta_pcr = put_delta_sum / call_delta_sum
            indicators[PCRType.DELTA_ADJUSTED] = PCRIndicator(
                symbol=symbol,
                timestamp=now,
                pcr_type=PCRType.DELTA_ADJUSTED,
                pcr_value=delta_pcr,
                put_volume=put_volume,
                call_volume=call_volume,
                put_oi=put_oi,
                call_oi=call_oi,
                interpretation=self._interpret_pcr(delta_pcr),
            )

        return indicators

    def _interpret_pcr(self, pcr: float) -> SentimentLevel:
        """解读PCR值"""
        if pcr < 0.5:
            return SentimentLevel.EXTREME_GREED
        elif pcr < 0.7:
            return SentimentLevel.GREED
        elif pcr < 1.0:
            return SentimentLevel.NEUTRAL
        elif pcr < 1.3:
            return SentimentLevel.FEAR
        else:
            return SentimentLevel.EXTREME_FEAR

    # ============ Unusual Activity Detection ============

    async def _detect_unusual_activity(
        self, symbol: str, option_chain: List[OptionData]
    ) -> List[UnusualActivity]:
        """检测异常活动"""
        activities = []

        for option in option_chain:
            # 获取历史数据计算平均成交量
            history = await self._option_provider.get_option_history(
                symbol, option.option_symbol, 20
            )
            avg_volume = sum(h.volume for h in history) / len(history) if history else 0

            # 检测大单交易
            if option.volume >= self._config.block_trade_min_contracts:
                activities.append(UnusualActivity(
                    symbol=symbol,
                    option_symbol=option.option_symbol,
                    activity_type=ActivityType.BLOCK_TRADE,
                    option_type=option.option_type,
                    strike=option.strike,
                    expiry=option.expiry,
                    volume=option.volume,
                    avg_volume=avg_volume,
                    volume_ratio=option.volume / avg_volume if avg_volume > 0 else 0,
                    oi_change=0,  # 需要历史数据计算
                    premium=option.last * option.volume,
                    notional_value=option.strike * option.volume,
                    flow_direction=self._determine_flow_direction(option),
                    confidence=0.8,
                ))

            # 检测异常成交量
            if avg_volume > 0 and option.volume > avg_volume * self._config.unusual_volume_threshold:
                activities.append(UnusualActivity(
                    symbol=symbol,
                    option_symbol=option.option_symbol,
                    activity_type=ActivityType.UNUSUAL_VOLUME,
                    option_type=option.option_type,
                    strike=option.strike,
                    expiry=option.expiry,
                    volume=option.volume,
                    avg_volume=avg_volume,
                    volume_ratio=option.volume / avg_volume,
                    oi_change=0,
                    premium=option.last * option.volume,
                    notional_value=option.strike * option.volume,
                    flow_direction=self._determine_flow_direction(option),
                    confidence=0.7,
                ))

            # 检测持仓增加
            if history:
                oi_change = option.open_interest - history[-1].open_interest
                if oi_change > option.open_interest * self._config.oi_buildup_threshold:
                    activities.append(UnusualActivity(
                        symbol=symbol,
                        option_symbol=option.option_symbol,
                        activity_type=ActivityType.OI_BUILDUP,
                        option_type=option.option_type,
                        strike=option.strike,
                        expiry=option.expiry,
                        volume=option.volume,
                        avg_volume=avg_volume,
                        volume_ratio=option.volume / avg_volume if avg_volume > 0 else 0,
                        oi_change=oi_change,
                        premium=option.last * oi_change,
                        notional_value=option.strike * oi_change,
                        flow_direction=self._determine_flow_direction(option),
                        confidence=0.75,
                    ))

        return activities

    def _determine_flow_direction(self, option: OptionData) -> FlowDirection:
        """判断期权流向"""
        # 简化实现：根据买卖价和最新价判断
        mid = (option.bid + option.ask) / 2
        if option.last > mid + 0.02:
            # 接近卖价，可能是买入
            return FlowDirection.BULLISH if option.option_type == OptionType.CALL else FlowDirection.BEARISH
        elif option.last < mid - 0.02:
            # 接近买价，可能是卖出
            return FlowDirection.BEARISH if option.option_type == OptionType.CALL else FlowDirection.BULLISH
        else:
            return FlowDirection.NEUTRAL

    # ============ Volatility Skew Calculation ============

    async def _calculate_volatility_skew(
        self, symbol: str, option_chain: List[OptionData]
    ) -> List[VolatilitySkew]:
        """计算波动率偏度"""
        skews = []
        now = datetime.now()

        # 按到期日分组
        expiry_groups: Dict[datetime, List[OptionData]] = {}
        for option in option_chain:
            if option.implied_vol is not None:
                if option.expiry not in expiry_groups:
                    expiry_groups[option.expiry] = []
                expiry_groups[option.expiry].append(option)

        # 计算每个到期日的偏度
        for expiry, options in expiry_groups.items():
            if len(options) < 5:  # 至少需要5个点
                continue

            # 找ATM行权价和波动率
            spot_price = options[0].spot_price
            atm_option = min(options, key=lambda o: abs(o.strike - spot_price))
            atm_strike = atm_option.strike
            atm_vol = atm_option.implied_vol or 0.2

            # 分离看涨和看跌
            calls = [o for o in options if o.option_type == OptionType.CALL]
            puts = [o for o in options if o.option_type == OptionType.PUT]

            # 计算25D和10D点的波动率
            put_25d_vol = self._find_delta_vol(puts, spot_price, 0.75)  # 约等于25D
            call_25d_vol = self._find_delta_vol(calls, spot_price, 1.25)
            put_10d_vol = self._find_delta_vol(puts, spot_price, 0.90)  # 约等于10D
            call_10d_vol = self._find_delta_vol(calls, spot_price, 1.10)

            # 计算风险逆转（Risk Reversal）
            rr_25d = call_25d_vol - put_25d_vol if call_25d_vol and put_25d_vol else 0
            rr_10d = call_10d_vol - put_10d_vol if call_10d_vol and put_10d_vol else 0

            # 计算蝶式（Butterfly）
            bf_25d = ((call_25d_vol + put_25d_vol) / 2 - atm_vol) if call_25d_vol and put_25d_vol else 0
            bf_10d = ((call_10d_vol + put_10d_vol) / 2 - atm_vol) if call_10d_vol and put_10d_vol else 0

            # 计算偏度指数
            skew_index = rr_25d  # 简化：使用25D RR作为偏度指数

            # 计算看跌和看涨偏度
            put_skew = self._calculate_wing_skew(puts, atm_strike, atm_vol)
            call_skew = self._calculate_wing_skew(calls, atm_strike, atm_vol)

            # 计算偏度斜率
            skew_slope = (call_skew - put_skew) / (2 * atm_strike) if atm_strike > 0 else 0

            # 计算微笑凸度
            smile_convexity = bf_25d

            skews.append(VolatilitySkew(
                symbol=symbol,
                expiry=expiry,
                timestamp=now,
                atm_strike=atm_strike,
                atm_vol=atm_vol,
                skew_index=skew_index,
                risk_reversal_25d=rr_25d,
                risk_reversal_10d=rr_10d,
                butterfly_25d=bf_25d,
                butterfly_10d=bf_10d,
                put_skew=put_skew,
                call_skew=call_skew,
                skew_slope=skew_slope,
                smile_convexity=smile_convexity,
            ))

        return skews

    def _find_delta_vol(
        self, options: List[OptionData], spot: float, moneyness: float
    ) -> Optional[float]:
        """查找指定货币性的波动率"""
        target_strike = spot * moneyness
        if not options:
            return None
        closest = min(options, key=lambda o: abs(o.strike - target_strike))
        return closest.implied_vol

    def _calculate_wing_skew(
        self, options: List[OptionData], atm_strike: float, atm_vol: float
    ) -> float:
        """计算翼部偏度"""
        if not options:
            return 0.0

        # 找到OTM期权
        otm_options = [o for o in options if abs(o.strike - atm_strike) > atm_strike * 0.05]
        if not otm_options:
            return 0.0

        # 计算平均偏度
        total_skew = sum(o.implied_vol - atm_vol for o in otm_options if o.implied_vol)
        return total_skew / len(otm_options)

    # ============ Sentiment Calculation ============

    async def _calculate_sentiment(
        self, symbol: str, option_chain: List[OptionData]
    ) -> OptionSentiment:
        """计算期权情绪"""
        now = datetime.now()
        spot_price = option_chain[0].spot_price if option_chain else 100.0

        # 计算聪明钱指数（Smart Money Index）
        smart_money_index = self._calculate_smart_money_index(option_chain)

        # 计算期权痛点（Option Pain）
        max_pain_strike, option_pain = self._calculate_option_pain(option_chain, spot_price)
        pain_distance = ((spot_price - max_pain_strike) / spot_price * 100) if spot_price > 0 else 0

        # 分离看涨和看跌
        calls = [o for o in option_chain if o.option_type == OptionType.CALL]
        puts = [o for o in option_chain if o.option_type == OptionType.PUT]

        # 计算看涨看跌比率
        call_volume = sum(c.volume for c in calls)
        put_volume = sum(p.volume for p in puts)
        call_put_ratio = call_volume / put_volume if put_volume > 0 else 1.0

        # 计算看跌看涨平价
        put_call_parity = self._calculate_put_call_parity(calls, puts, spot_price)

        # 计算流入
        bullish_flow = sum(c.last * c.volume for c in calls if c.last > (c.bid + c.ask) / 2)
        bearish_flow = sum(p.last * p.volume for p in puts if p.last > (p.bid + p.ask) / 2)

        # 计算情绪分数（-100到100）
        sentiment_score = self._calculate_sentiment_score(
            call_put_ratio, smart_money_index, pain_distance, bullish_flow, bearish_flow
        )

        # 判断情绪水平
        sentiment_level = self._classify_sentiment(sentiment_score)

        return OptionSentiment(
            symbol=symbol,
            timestamp=now,
            smart_money_index=smart_money_index,
            option_pain=option_pain,
            max_pain_strike=max_pain_strike,
            pain_distance=pain_distance,
            call_put_ratio=call_put_ratio,
            put_call_parity=put_call_parity,
            sentiment_score=sentiment_score,
            sentiment_level=sentiment_level,
            bullish_flow=bullish_flow,
            bearish_flow=bearish_flow,
        )

    def _calculate_smart_money_index(self, option_chain: List[OptionData]) -> float:
        """计算聪明钱指数"""
        # 简化实现：根据大单和持仓量计算
        total_premium = sum(o.last * o.open_interest for o in option_chain if o.last and o.open_interest)
        total_volume_premium = sum(o.last * o.volume for o in option_chain if o.last and o.volume)

        if total_volume_premium == 0:
            return 50.0

        # 持仓量溢价比率
        ratio = total_premium / total_volume_premium
        # 归一化到0-100
        return min(max(ratio * 50, 0), 100)

    def _calculate_option_pain(
        self, option_chain: List[OptionData], spot_price: float
    ) -> Tuple[float, float]:
        """计算期权痛点和最大痛点行权价"""
        if not option_chain:
            return spot_price, 0.0

        # 获取所有行权价
        strikes = sorted(set(o.strike for o in option_chain))
        if not strikes:
            return spot_price, 0.0

        # 计算每个行权价的总损失
        max_pain_strike = strikes[0]
        min_pain = float('inf')

        for strike in strikes:
            # 计算在该行权价到期时，所有期权持有者的总损失
            total_loss = 0.0

            for option in option_chain:
                if option.option_type == OptionType.CALL:
                    # 看涨期权：行权价低于当前价才有价值
                    intrinsic = max(strike - option.strike, 0)
                else:
                    # 看跌期权：行权价高于当前价才有价值
                    intrinsic = max(option.strike - strike, 0)

                total_loss += intrinsic * option.open_interest

            if total_loss < min_pain:
                min_pain = total_loss
                max_pain_strike = strike

        return max_pain_strike, min_pain

    def _calculate_put_call_parity(
        self, calls: List[OptionData], puts: List[OptionData], spot_price: float
    ) -> float:
        """计算看跌看涨平价"""
        # 简化实现：比较相同行权价的看涨和看跌期权价格
        parity_sum = 0.0
        count = 0

        for call in calls:
            # 找到相同行权价和到期日的看跌期权
            matching_puts = [
                p for p in puts
                if abs(p.strike - call.strike) < 0.01 and p.expiry == call.expiry
            ]
            if matching_puts:
                put = matching_puts[0]
                # 看跌看涨平价：C - P = S - K * e^(-rT)
                # 简化：C - P ≈ S - K
                parity = (call.last - put.last) - (spot_price - call.strike)
                parity_sum += abs(parity)
                count += 1

        return parity_sum / count if count > 0 else 0.0

    def _calculate_sentiment_score(
        self,
        call_put_ratio: float,
        smart_money_index: float,
        pain_distance: float,
        bullish_flow: float,
        bearish_flow: float,
    ) -> float:
        """计算情绪分数"""
        # 综合多个因素计算情绪分数
        score = 0.0

        # CPR贡献（权重30%）
        if call_put_ratio > 1.5:
            score += 30  # 看涨
        elif call_put_ratio > 1.0:
            score += 15
        elif call_put_ratio < 0.7:
            score -= 30  # 看跌
        elif call_put_ratio < 1.0:
            score -= 15

        # 聪明钱指数贡献（权重20%）
        score += (smart_money_index - 50) * 0.4

        # 痛点距离贡献（权重20%）
        score += pain_distance * 0.2

        # 流入流出贡献（权重30%）
        total_flow = bullish_flow + bearish_flow
        if total_flow > 0:
            flow_ratio = (bullish_flow - bearish_flow) / total_flow
            score += flow_ratio * 30

        return max(min(score, 100), -100)

    def _classify_sentiment(self, sentiment_score: float) -> SentimentLevel:
        """分类情绪水平"""
        if sentiment_score > 80:
            return SentimentLevel.EXTREME_GREED
        elif sentiment_score > 40:
            return SentimentLevel.GREED
        elif sentiment_score > -40:
            return SentimentLevel.NEUTRAL
        elif sentiment_score > -80:
            return SentimentLevel.FEAR
        else:
            return SentimentLevel.EXTREME_FEAR

    # ============ Term Structure Calculation ============

    async def _calculate_term_structure(self, symbol: str) -> Optional[TermStructure]:
        """计算期限结构"""
        try:
            # 获取期货链
            futures_chain = await self._futures_provider.get_futures_chain(symbol)
            if len(futures_chain) < 2:
                return None

            # 获取现货价格
            spot_price = await self._futures_provider.get_spot_price(symbol)

            # 按到期日排序
            sorted_contracts = sorted(futures_chain, key=lambda c: c.expiry)

            # 计算前后月价格
            front_month = sorted_contracts[0]
            back_month = sorted_contracts[-1]

            # 判断形态
            shape = self._classify_term_structure_shape(sorted_contracts)

            # 计算升贴水百分比
            contango_pct = ((front_month.price - spot_price) / spot_price * 100) if spot_price > 0 else 0

            # 计算展期收益（Roll Yield）
            if len(sorted_contracts) >= 2:
                time_diff = (sorted_contracts[1].time_to_expiry - sorted_contracts[0].time_to_expiry)
                price_diff = sorted_contracts[1].price - sorted_contracts[0].price
                roll_yield = (price_diff / sorted_contracts[0].price / time_diff * 365) if time_diff > 0 else 0
            else:
                roll_yield = 0.0

            # 计算期限溢价（Term Premium）
            term_premium = (back_month.price - front_month.price) / front_month.price if front_month.price > 0 else 0

            # 计算便利收益率（Convenience Yield）
            # 简化：根据基差计算
            convenience_yield = -contango_pct / 100 if contango_pct < 0 else 0

            # 计算持有成本
            carrying_cost = contango_pct / 100 if contango_pct > 0 else 0

            return TermStructure(
                symbol=symbol,
                timestamp=datetime.now(),
                spot_price=spot_price,
                contracts=sorted_contracts,
                shape=shape,
                front_month_price=front_month.price,
                back_month_price=back_month.price,
                contango_pct=contango_pct,
                roll_yield=roll_yield,
                term_premium=term_premium,
                convenience_yield=convenience_yield,
                carrying_cost=carrying_cost,
            )

        except Exception as e:
            logger.error(f"Error calculating term structure for {symbol}: {e}")
            return None

    def _classify_term_structure_shape(
        self, contracts: List[FuturesContract]
    ) -> TermStructureShape:
        """分类期限结构形态"""
        if len(contracts) < 2:
            return TermStructureShape.FLAT

        prices = [c.price for c in contracts]

        # 检查是否单调递增或递减
        is_increasing = all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))
        is_decreasing = all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))

        if is_increasing:
            slope = (prices[-1] - prices[0]) / prices[0]
            if slope > self._config.contango_threshold:
                return TermStructureShape.CONTANGO
            else:
                return TermStructureShape.FLAT
        elif is_decreasing:
            slope = (prices[0] - prices[-1]) / prices[0]
            if slope > -self._config.backwardation_threshold:
                return TermStructureShape.BACKWARDATION
            else:
                return TermStructureShape.INVERTED
        else:
            # 检查是否有驼峰
            max_idx = prices.index(max(prices))
            if 0 < max_idx < len(prices) - 1:
                return TermStructureShape.HUMPED
            else:
                return TermStructureShape.FLAT

    # ============ Alert Detection ============

    def _check_alerts(
        self, symbol: str, snapshot: DerivativesSnapshot
    ) -> List[DerivativesAlert]:
        """检查警报"""
        alerts = []

        # 检查PCR极值
        for pcr_type, pcr in snapshot.pcr_indicators.items():
            if pcr.pcr_value < self._config.pcr_extreme_low:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    alert_type=AlertType.PCR_EXTREME,
                    severity=AlertSeverity.HIGH,
                    message=f"Extremely low {pcr_type.value} PCR: {pcr.pcr_value:.2f}",
                    details={"pcr_type": pcr_type.value, "value": pcr.pcr_value},
                ))
            elif pcr.pcr_value > self._config.pcr_extreme_high:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    alert_type=AlertType.PCR_EXTREME,
                    severity=AlertSeverity.HIGH,
                    message=f"Extremely high {pcr_type.value} PCR: {pcr.pcr_value:.2f}",
                    details={"pcr_type": pcr_type.value, "value": pcr.pcr_value},
                ))

        # 检查异常活动
        if len(snapshot.unusual_activities) > 5:
            alerts.append(self._create_alert(
                symbol=symbol,
                alert_type=AlertType.UNUSUAL_ACTIVITY,
                severity=AlertSeverity.MEDIUM,
                message=f"Multiple unusual activities detected: {len(snapshot.unusual_activities)}",
                details={"count": len(snapshot.unusual_activities)},
            ))

        # 检查偏度异常
        for skew in snapshot.volatility_skews:
            if abs(skew.skew_index) > self._config.skew_anomaly_threshold:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    alert_type=AlertType.SKEW_ANOMALY,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Volatility skew anomaly: {skew.skew_index:.4f}",
                    details={"skew_index": skew.skew_index, "expiry": skew.expiry.isoformat()},
                ))

        # 检查极端情绪
        if abs(snapshot.sentiment.sentiment_score) > self._config.sentiment_extreme_threshold:
            severity = AlertSeverity.CRITICAL if abs(snapshot.sentiment.sentiment_score) > 90 else AlertSeverity.HIGH
            alerts.append(self._create_alert(
                symbol=symbol,
                alert_type=AlertType.SENTIMENT_SHIFT,
                severity=severity,
                message=f"Extreme sentiment: {snapshot.sentiment.sentiment_level.value}",
                details={
                    "sentiment_score": snapshot.sentiment.sentiment_score,
                    "sentiment_level": snapshot.sentiment.sentiment_level.value,
                },
            ))

        # 检查期限结构变化
        if snapshot.term_structure:
            if snapshot.term_structure.shape in [TermStructureShape.INVERTED, TermStructureShape.BACKWARDATION]:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    alert_type=AlertType.TERM_STRUCTURE_CHANGE,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Term structure in {snapshot.term_structure.shape.value}",
                    details={
                        "shape": snapshot.term_structure.shape.value,
                        "contango_pct": snapshot.term_structure.contango_pct,
                    },
                ))

        # 检查期权痛点
        pain_distance = abs(snapshot.sentiment.pain_distance)
        if pain_distance > 5.0:  # 距离痛点超过5%
            alerts.append(self._create_alert(
                symbol=symbol,
                alert_type=AlertType.OPTION_PAIN,
                severity=AlertSeverity.LOW,
                message=f"Price {pain_distance:.2f}% away from max pain strike",
                details={
                    "max_pain_strike": snapshot.sentiment.max_pain_strike,
                    "pain_distance": snapshot.sentiment.pain_distance,
                },
            ))

        return alerts

    def _create_alert(
        self,
        symbol: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any],
    ) -> DerivativesAlert:
        """创建警报"""
        self._alert_counter += 1
        return DerivativesAlert(
            alert_id=f"DERIV_ALERT_{self._alert_counter}",
            symbol=symbol,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
        )

    # ============ History Management ============

    def _add_to_history(self, symbol: str, snapshot: DerivativesSnapshot) -> None:
        """添加到历史记录"""
        if symbol not in self._snapshot_history:
            self._snapshot_history[symbol] = []

        self._snapshot_history[symbol].append(snapshot)

        # 限制大小
        if len(self._snapshot_history[symbol]) > self._config.history_size:
            self._snapshot_history[symbol] = self._snapshot_history[symbol][-self._config.history_size:]

    # ============ Callbacks ============

    def on_snapshot_update(
        self, callback: Callable[[DerivativesSnapshot], None]
    ) -> None:
        """注册快照更新回调"""
        self._snapshot_callbacks.append(callback)

    def on_alert(
        self, callback: Callable[[DerivativesAlert], None]
    ) -> None:
        """注册警报回调"""
        self._alert_callbacks.append(callback)

    def _notify_snapshot(self, snapshot: DerivativesSnapshot) -> None:
        """通知快照更新"""
        for callback in self._snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in snapshot callback: {e}")

    def _notify_alert(self, alert: DerivativesAlert) -> None:
        """通知警报"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    # ============ Public API ============

    def get_pcr(self, symbol: str, pcr_type: PCRType) -> Optional[PCRIndicator]:
        """获取PCR指标"""
        if symbol in self._pcr_cache:
            return self._pcr_cache[symbol].get(pcr_type)
        return None

    def get_all_pcr(self, symbol: str) -> Dict[PCRType, PCRIndicator]:
        """获取所有PCR指标"""
        return self._pcr_cache.get(symbol, {})

    def get_unusual_activities(self, symbol: str) -> List[UnusualActivity]:
        """获取异常活动"""
        return self._activities_cache.get(symbol, [])

    def get_volatility_skews(self, symbol: str) -> List[VolatilitySkew]:
        """获取波动率偏度"""
        return self._skew_cache.get(symbol, [])

    def get_sentiment(self, symbol: str) -> Optional[OptionSentiment]:
        """获取情绪指标"""
        return self._sentiment_cache.get(symbol)

    def get_term_structure(self, symbol: str) -> Optional[TermStructure]:
        """获取期限结构"""
        return self._term_structure_cache.get(symbol)

    def get_snapshot(self, symbol: str) -> Optional[DerivativesSnapshot]:
        """获取最新快照"""
        history = self._snapshot_history.get(symbol, [])
        return history[-1] if history else None

    def get_snapshot_history(self, symbol: str, limit: int = 100) -> List[DerivativesSnapshot]:
        """获取快照历史"""
        history = self._snapshot_history.get(symbol, [])
        return history[-limit:]

    def get_alerts(self, limit: int = 100) -> List[DerivativesAlert]:
        """获取警报列表"""
        return self._alert_history[-limit:]

    def get_state(self) -> MonitorState:
        """获取监控器状态"""
        return self._state

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "symbols_tracked": len(self._symbols),
            "total_alerts": len(self._alert_history),
            "symbols": {
                symbol: {
                    "pcr_count": len(self._pcr_cache.get(symbol, {})),
                    "unusual_activities": len(self._activities_cache.get(symbol, [])),
                    "volatility_skews": len(self._skew_cache.get(symbol, [])),
                    "has_sentiment": symbol in self._sentiment_cache,
                    "has_term_structure": symbol in self._term_structure_cache,
                    "snapshot_history_size": len(self._snapshot_history.get(symbol, [])),
                }
                for symbol in self._symbols
            },
        }


# ============ Factory Functions ============


def create_derivatives_indicators(
    update_interval_ms: int = 5000,
    pcr_extreme_low: float = 0.5,
    pcr_extreme_high: float = 1.5,
    block_trade_min_contracts: int = 100,
    unusual_volume_threshold: float = 3.0,
    option_provider: Optional[OptionDataProvider] = None,
    futures_provider: Optional[FuturesDataProvider] = None,
) -> DerivativesIndicators:
    """创建衍生品指标监控器"""
    config = DerivativesConfig(
        update_interval_ms=update_interval_ms,
        pcr_extreme_low=pcr_extreme_low,
        pcr_extreme_high=pcr_extreme_high,
        block_trade_min_contracts=block_trade_min_contracts,
        unusual_volume_threshold=unusual_volume_threshold,
    )
    return DerivativesIndicators(
        config=config,
        option_provider=option_provider,
        futures_provider=futures_provider,
    )


def create_option_data(
    symbol: str,
    option_symbol: str,
    option_type: str,
    strike: float,
    expiry: datetime,
    spot_price: float,
    volume: int,
    open_interest: int,
    bid: float,
    ask: float,
    last: float,
    implied_vol: Optional[float] = None,
) -> OptionData:
    """创建期权数据"""
    return OptionData(
        symbol=symbol,
        option_symbol=option_symbol,
        option_type=OptionType(option_type),
        strike=strike,
        expiry=expiry,
        spot_price=spot_price,
        volume=volume,
        open_interest=open_interest,
        bid=bid,
        ask=ask,
        last=last,
        implied_vol=implied_vol,
    )


def create_futures_contract(
    symbol: str,
    contract_month: str,
    expiry: datetime,
    price: float,
    volume: int,
    open_interest: int,
    spot_price: float,
) -> FuturesContract:
    """创建期货合约"""
    time_to_expiry = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
    basis = price - spot_price
    return FuturesContract(
        symbol=symbol,
        contract_month=contract_month,
        expiry=expiry,
        price=price,
        volume=volume,
        open_interest=open_interest,
        basis=basis,
        time_to_expiry=time_to_expiry,
    )


async def quick_pcr_check(
    indicators: DerivativesIndicators,
    symbol: str,
) -> Dict[str, float]:
    """快速PCR检查"""
    all_pcr = indicators.get_all_pcr(symbol)
    return {
        pcr_type.value: pcr.pcr_value
        for pcr_type, pcr in all_pcr.items()
    }


# ============ Exports ============


__all__ = [
    # Enums
    "OptionType",
    "PCRType",
    "FlowDirection",
    "ActivityType",
    "TermStructureShape",
    "SentimentLevel",
    "AlertSeverity",
    "AlertType",
    "MonitorState",
    # Data Classes
    "OptionData",
    "PCRIndicator",
    "UnusualActivity",
    "VolatilitySkew",
    "OptionSentiment",
    "FuturesContract",
    "TermStructure",
    "DerivativesAlert",
    "DerivativesSnapshot",
    "DerivativesConfig",
    # Interfaces
    "OptionDataProvider",
    "FuturesDataProvider",
    # Simulated Providers
    "SimulatedOptionDataProvider",
    "SimulatedFuturesDataProvider",
    # Main Class
    "DerivativesIndicators",
    # Factory Functions
    "create_derivatives_indicators",
    "create_option_data",
    "create_futures_contract",
    "quick_pcr_check",
]
