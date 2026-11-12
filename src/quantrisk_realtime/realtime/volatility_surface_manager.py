"""
Volatility Surface Manager - 波动率曲面管理器

实时波动率曲面构建、插值和监控系统。
支持期权定价、风险管理和波动率交易策略。

核心功能：
1. 波动率曲面构建 - 从期权市场数据构建隐含波动率曲面
2. 曲面插值 - 多种插值方法支持任意行权价/到期日查询
3. 波动率微笑分析 - 分析不同到期日的波动率微笑形态
4. 期限结构分析 - 分析波动率期限结构
5. 曲面动态监控 - 实时监控曲面变化和异常
6. 历史波动率对比 - 隐含vs实现波动率分析
7. 波动率套利检测 - 检测曲面套利机会
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============


class OptionType(str, Enum):
    """期权类型"""
    CALL = "call"
    PUT = "put"


class InterpolationMethod(str, Enum):
    """插值方法"""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SABR = "sabr"
    SVI = "svi"
    FLAT = "flat"


class SmileShape(str, Enum):
    """波动率微笑形态"""
    SMILE = "smile"           # 标准微笑（两端高）
    SMIRK = "smirk"           # 歪斜微笑（一端高）
    FROWN = "frown"           # 皱眉（两端低）
    FLAT = "flat"             # 平坦
    SKEW_LEFT = "skew_left"   # 左偏
    SKEW_RIGHT = "skew_right" # 右偏


class TermStructureShape(str, Enum):
    """期限结构形态"""
    NORMAL = "normal"           # 正常（远期高）
    INVERTED = "inverted"       # 倒挂（近期高）
    FLAT = "flat"               # 平坦
    HUMPED = "humped"           # 驼峰
    U_SHAPED = "u_shaped"       # U形


class AlertType(str, Enum):
    """警报类型"""
    SURFACE_SHIFT = "surface_shift"           # 曲面整体移动
    SMILE_CHANGE = "smile_change"             # 微笑形态变化
    TERM_STRUCTURE_CHANGE = "term_structure"  # 期限结构变化
    ARBITRAGE_DETECTED = "arbitrage"          # 检测到套利
    DATA_QUALITY = "data_quality"             # 数据质量问题
    EXTREME_VOL = "extreme_vol"               # 极端波动率


class SurfaceState(str, Enum):
    """曲面状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    UPDATING = "updating"
    STALE = "stale"
    ERROR = "error"


# ============ Data Classes ============


@dataclass
class OptionQuote:
    """期权报价"""
    symbol: str                    # 标的代码
    option_symbol: str             # 期权代码
    option_type: OptionType        # 期权类型
    strike: float                  # 行权价
    expiry: datetime               # 到期日
    bid: float                     # 买价
    ask: float                     # 卖价
    mid: float                     # 中间价
    last: float                    # 最新价
    volume: int                    # 成交量
    open_interest: int             # 持仓量
    implied_vol: Optional[float]   # 隐含波动率
    delta: Optional[float] = None  # Delta
    gamma: Optional[float] = None  # Gamma
    theta: Optional[float] = None  # Theta
    vega: Optional[float] = None   # Vega
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_vol": self.implied_vol,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VolatilityPoint:
    """波动率点"""
    strike: float           # 行权价
    expiry: datetime        # 到期日
    implied_vol: float      # 隐含波动率
    moneyness: float        # 货币性（K/S）
    time_to_expiry: float   # 到期时间（年）
    log_moneyness: float    # 对数货币性
    forward_price: float    # 远期价格
    option_type: OptionType # 期权类型
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    confidence: float = 1.0  # 置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "implied_vol": self.implied_vol,
            "moneyness": self.moneyness,
            "time_to_expiry": self.time_to_expiry,
            "log_moneyness": self.log_moneyness,
            "forward_price": self.forward_price,
            "option_type": self.option_type.value,
            "bid_vol": self.bid_vol,
            "ask_vol": self.ask_vol,
            "confidence": self.confidence,
        }


@dataclass
class VolatilitySlice:
    """波动率切片（固定到期日）"""
    expiry: datetime
    time_to_expiry: float
    atm_vol: float                         # 平值波动率
    points: List[VolatilityPoint]          # 所有点
    smile_shape: SmileShape                # 微笑形态
    skew: float                            # 偏度（25D RR）
    butterfly: float                       # 蝶式（25D BF）
    atm_strike: float                      # 平值行权价
    forward_price: float                   # 远期价格
    put_25d_vol: Optional[float] = None    # 25D看跌波动率
    call_25d_vol: Optional[float] = None   # 25D看涨波动率
    put_10d_vol: Optional[float] = None    # 10D看跌波动率
    call_10d_vol: Optional[float] = None   # 10D看涨波动率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expiry": self.expiry.isoformat(),
            "time_to_expiry": self.time_to_expiry,
            "atm_vol": self.atm_vol,
            "points": [p.to_dict() for p in self.points],
            "smile_shape": self.smile_shape.value,
            "skew": self.skew,
            "butterfly": self.butterfly,
            "atm_strike": self.atm_strike,
            "forward_price": self.forward_price,
            "put_25d_vol": self.put_25d_vol,
            "call_25d_vol": self.call_25d_vol,
            "put_10d_vol": self.put_10d_vol,
            "call_10d_vol": self.call_10d_vol,
        }


@dataclass
class VolatilitySurface:
    """波动率曲面"""
    symbol: str
    spot_price: float
    timestamp: datetime
    slices: List[VolatilitySlice]
    term_structure_shape: TermStructureShape
    atm_term_structure: List[Tuple[float, float]]  # (time_to_expiry, vol)
    overall_level: float                            # 整体水平
    average_skew: float                             # 平均偏度
    average_butterfly: float                        # 平均蝶式
    interpolation_method: InterpolationMethod
    state: SurfaceState = SurfaceState.READY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "timestamp": self.timestamp.isoformat(),
            "slices": [s.to_dict() for s in self.slices],
            "term_structure_shape": self.term_structure_shape.value,
            "atm_term_structure": self.atm_term_structure,
            "overall_level": self.overall_level,
            "average_skew": self.average_skew,
            "average_butterfly": self.average_butterfly,
            "interpolation_method": self.interpolation_method.value,
            "state": self.state.value,
        }

    def get_vol(self, strike: float, time_to_expiry: float) -> Optional[float]:
        """获取指定行权价和到期时间的波动率"""
        # 找到最近的两个切片进行插值
        if not self.slices:
            return None

        # 按到期时间排序
        sorted_slices = sorted(self.slices, key=lambda s: s.time_to_expiry)

        # 找到包围目标时间的切片
        lower_slice = None
        upper_slice = None

        for i, slice_ in enumerate(sorted_slices):
            if slice_.time_to_expiry <= time_to_expiry:
                lower_slice = slice_
            if slice_.time_to_expiry >= time_to_expiry and upper_slice is None:
                upper_slice = slice_

        if lower_slice is None and upper_slice is None:
            return None

        # 如果只有一个切片，直接在该切片上插值
        if lower_slice is None:
            return self._interpolate_strike(upper_slice, strike)
        if upper_slice is None:
            return self._interpolate_strike(lower_slice, strike)
        if lower_slice == upper_slice:
            return self._interpolate_strike(lower_slice, strike)

        # 在两个切片上分别插值，然后时间插值
        lower_vol = self._interpolate_strike(lower_slice, strike)
        upper_vol = self._interpolate_strike(upper_slice, strike)

        if lower_vol is None or upper_vol is None:
            return lower_vol or upper_vol

        # 时间线性插值
        t_lower = lower_slice.time_to_expiry
        t_upper = upper_slice.time_to_expiry
        weight = (time_to_expiry - t_lower) / (t_upper - t_lower) if t_upper != t_lower else 0

        return lower_vol + weight * (upper_vol - lower_vol)

    def _interpolate_strike(self, slice_: VolatilitySlice, strike: float) -> Optional[float]:
        """在切片上进行行权价插值"""
        if not slice_.points:
            return slice_.atm_vol

        # 按行权价排序
        sorted_points = sorted(slice_.points, key=lambda p: p.strike)

        # 找到包围目标行权价的点
        lower_point = None
        upper_point = None

        for point in sorted_points:
            if point.strike <= strike:
                lower_point = point
            if point.strike >= strike and upper_point is None:
                upper_point = point

        if lower_point is None and upper_point is None:
            return slice_.atm_vol
        if lower_point is None:
            return upper_point.implied_vol
        if upper_point is None:
            return lower_point.implied_vol
        if lower_point == upper_point:
            return lower_point.implied_vol

        # 线性插值
        k_lower = lower_point.strike
        k_upper = upper_point.strike
        weight = (strike - k_lower) / (k_upper - k_lower) if k_upper != k_lower else 0

        return lower_point.implied_vol + weight * (upper_point.implied_vol - lower_point.implied_vol)


@dataclass
class SurfaceChange:
    """曲面变化"""
    symbol: str
    timestamp: datetime
    previous_level: float
    current_level: float
    level_change: float
    level_change_pct: float
    skew_change: float
    butterfly_change: float
    term_structure_change: float
    smile_shape_changed: bool
    term_structure_shape_changed: bool
    max_point_change: float
    max_point_strike: float
    max_point_expiry: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "previous_level": self.previous_level,
            "current_level": self.current_level,
            "level_change": self.level_change,
            "level_change_pct": self.level_change_pct,
            "skew_change": self.skew_change,
            "butterfly_change": self.butterfly_change,
            "term_structure_change": self.term_structure_change,
            "smile_shape_changed": self.smile_shape_changed,
            "term_structure_shape_changed": self.term_structure_shape_changed,
            "max_point_change": self.max_point_change,
            "max_point_strike": self.max_point_strike,
            "max_point_expiry": self.max_point_expiry.isoformat(),
        }


@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    symbol: str
    arbitrage_type: str           # calendar, butterfly, vertical, etc.
    description: str
    strikes: List[float]
    expiries: List[datetime]
    implied_profit: float         # 理论利润
    bid_ask_cost: float           # 买卖价差成本
    net_profit: float             # 净利润
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "arbitrage_type": self.arbitrage_type,
            "description": self.description,
            "strikes": self.strikes,
            "expiries": [e.isoformat() for e in self.expiries],
            "implied_profit": self.implied_profit,
            "bid_ask_cost": self.bid_ask_cost,
            "net_profit": self.net_profit,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VolatilityAlert:
    """波动率警报"""
    symbol: str
    alert_type: AlertType
    severity: str  # low, medium, high, critical
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HistoricalVolComparison:
    """历史波动率对比"""
    symbol: str
    timestamp: datetime
    implied_vol: float            # 隐含波动率
    realized_vol_1d: float        # 1日实现波动率
    realized_vol_5d: float        # 5日实现波动率
    realized_vol_20d: float       # 20日实现波动率
    realized_vol_60d: float       # 60日实现波动率
    vol_premium: float            # 波动率溢价
    vol_premium_percentile: float # 溢价百分位

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "implied_vol": self.implied_vol,
            "realized_vol_1d": self.realized_vol_1d,
            "realized_vol_5d": self.realized_vol_5d,
            "realized_vol_20d": self.realized_vol_20d,
            "realized_vol_60d": self.realized_vol_60d,
            "vol_premium": self.vol_premium,
            "vol_premium_percentile": self.vol_premium_percentile,
        }


@dataclass
class SurfaceManagerConfig:
    """曲面管理器配置"""
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    min_points_per_slice: int = 3          # 每个切片最少点数
    max_time_to_expiry: float = 2.0        # 最大到期时间（年）
    min_time_to_expiry: float = 1/365      # 最小到期时间（1天）
    stale_threshold_seconds: int = 300     # 数据过期阈值
    level_change_alert_threshold: float = 0.02   # 水平变化警报阈值
    skew_change_alert_threshold: float = 0.01    # 偏度变化警报阈值
    extreme_vol_low: float = 0.05          # 极低波动率阈值
    extreme_vol_high: float = 1.5          # 极高波动率阈值
    arbitrage_min_profit: float = 0.001    # 套利最小利润
    update_interval_ms: int = 1000         # 更新间隔
    history_size: int = 1000               # 历史记录大小


# ============ Abstract Interfaces ============


class OptionDataProvider(ABC):
    """期权数据提供者接口"""

    @abstractmethod
    async def get_option_chain(self, symbol: str) -> List[OptionQuote]:
        """获取期权链"""
        pass

    @abstractmethod
    async def get_spot_price(self, symbol: str) -> float:
        """获取现货价格"""
        pass

    @abstractmethod
    async def get_risk_free_rate(self) -> float:
        """获取无风险利率"""
        pass

    @abstractmethod
    async def get_dividend_yield(self, symbol: str) -> float:
        """获取股息率"""
        pass


class HistoricalVolProvider(ABC):
    """历史波动率提供者接口"""

    @abstractmethod
    async def get_realized_volatility(
        self, symbol: str, window: int
    ) -> float:
        """获取实现波动率"""
        pass

    @abstractmethod
    async def get_volatility_history(
        self, symbol: str, days: int
    ) -> List[Tuple[datetime, float]]:
        """获取波动率历史"""
        pass


# ============ Simulated Providers ============


class SimulatedOptionDataProvider(OptionDataProvider):
    """模拟期权数据提供者"""

    def __init__(self):
        self._option_chains: Dict[str, List[OptionQuote]] = {}
        self._spot_prices: Dict[str, float] = {}
        self._risk_free_rate: float = 0.05
        self._dividend_yields: Dict[str, float] = {}

    def set_option_chain(self, symbol: str, quotes: List[OptionQuote]) -> None:
        """设置期权链"""
        self._option_chains[symbol] = quotes

    def set_spot_price(self, symbol: str, price: float) -> None:
        """设置现货价格"""
        self._spot_prices[symbol] = price

    def set_risk_free_rate(self, rate: float) -> None:
        """设置无风险利率"""
        self._risk_free_rate = rate

    def set_dividend_yield(self, symbol: str, yield_: float) -> None:
        """设置股息率"""
        self._dividend_yields[symbol] = yield_

    async def get_option_chain(self, symbol: str) -> List[OptionQuote]:
        return self._option_chains.get(symbol, [])

    async def get_spot_price(self, symbol: str) -> float:
        return self._spot_prices.get(symbol, 100.0)

    async def get_risk_free_rate(self) -> float:
        return self._risk_free_rate

    async def get_dividend_yield(self, symbol: str) -> float:
        return self._dividend_yields.get(symbol, 0.0)


class SimulatedHistoricalVolProvider(HistoricalVolProvider):
    """模拟历史波动率提供者"""

    def __init__(self):
        self._realized_vols: Dict[str, Dict[int, float]] = {}
        self._vol_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def set_realized_volatility(
        self, symbol: str, window: int, vol: float
    ) -> None:
        """设置实现波动率"""
        if symbol not in self._realized_vols:
            self._realized_vols[symbol] = {}
        self._realized_vols[symbol][window] = vol

    def set_volatility_history(
        self, symbol: str, history: List[Tuple[datetime, float]]
    ) -> None:
        """设置波动率历史"""
        self._vol_history[symbol] = history

    async def get_realized_volatility(
        self, symbol: str, window: int
    ) -> float:
        if symbol in self._realized_vols:
            return self._realized_vols[symbol].get(window, 0.2)
        return 0.2

    async def get_volatility_history(
        self, symbol: str, days: int
    ) -> List[Tuple[datetime, float]]:
        history = self._vol_history.get(symbol, [])
        return history[-days:] if history else []


# ============ Main Class ============


class VolatilitySurfaceManager:
    """波动率曲面管理器"""

    def __init__(
        self,
        config: SurfaceManagerConfig,
        option_provider: Optional[OptionDataProvider] = None,
        historical_provider: Optional[HistoricalVolProvider] = None,
    ):
        self._config = config
        self._option_provider = option_provider or simulated_default(SimulatedOptionDataProvider, "data from VolatilitySurfaceManager's simulated option_provider (SimulatedOptionDataProvider)")
        self._historical_provider = historical_provider or simulated_default(SimulatedHistoricalVolProvider, "data from VolatilitySurfaceManager's simulated historical_provider (SimulatedHistoricalVolProvider)")

        # 状态
        self._surfaces: Dict[str, VolatilitySurface] = {}
        self._previous_surfaces: Dict[str, VolatilitySurface] = {}
        self._symbols: Set[str] = set()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 历史记录
        self._surface_history: Dict[str, List[VolatilitySurface]] = {}
        self._change_history: Dict[str, List[SurfaceChange]] = {}
        self._alert_history: List[VolatilityAlert] = []
        self._arbitrage_history: List[ArbitrageOpportunity] = []

        # 回调
        self._surface_callbacks: List[Callable[[VolatilitySurface], None]] = []
        self._change_callbacks: List[Callable[[SurfaceChange], None]] = []
        self._alert_callbacks: List[Callable[[VolatilityAlert], None]] = []
        self._arbitrage_callbacks: List[Callable[[ArbitrageOpportunity], None]] = []

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动管理器"""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("VolatilitySurfaceManager started")

    async def stop(self) -> None:
        """停止管理器"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("VolatilitySurfaceManager stopped")

    async def _update_loop(self) -> None:
        """更新循环"""
        while self._running:
            try:
                for symbol in self._symbols:
                    await self._update_surface(symbol)

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
        self._surface_history[symbol] = []
        self._change_history[symbol] = []

    def remove_symbol(self, symbol: str) -> None:
        """移除监控标的"""
        self._symbols.discard(symbol)
        self._surfaces.pop(symbol, None)
        self._previous_surfaces.pop(symbol, None)

    def get_symbols(self) -> List[str]:
        """获取监控标的列表"""
        return list(self._symbols)

    # ============ Surface Building ============

    async def _update_surface(self, symbol: str) -> None:
        """更新曲面"""
        try:
            # 获取数据
            option_chain = await self._option_provider.get_option_chain(symbol)
            spot_price = await self._option_provider.get_spot_price(symbol)
            risk_free_rate = await self._option_provider.get_risk_free_rate()
            dividend_yield = await self._option_provider.get_dividend_yield(symbol)

            if not option_chain:
                return

            # 保存旧曲面
            if symbol in self._surfaces:
                self._previous_surfaces[symbol] = self._surfaces[symbol]

            # 构建新曲面
            surface = await self._build_surface(
                symbol, option_chain, spot_price, risk_free_rate, dividend_yield
            )

            self._surfaces[symbol] = surface

            # 记录历史
            self._add_to_history(symbol, surface)

            # 检测变化
            if symbol in self._previous_surfaces:
                change = self._detect_change(
                    symbol, self._previous_surfaces[symbol], surface
                )
                if change:
                    self._change_history[symbol].append(change)
                    self._notify_change(change)

                    # 检查是否需要警报
                    alerts = self._check_alerts(symbol, change, surface)
                    for alert in alerts:
                        self._alert_history.append(alert)
                        self._notify_alert(alert)

            # 检测套利机会
            arbitrage = await self._detect_arbitrage(symbol, surface)
            for opp in arbitrage:
                self._arbitrage_history.append(opp)
                self._notify_arbitrage(opp)

            # 通知曲面更新
            self._notify_surface(surface)

        except Exception as e:
            logger.error(f"Error updating surface for {symbol}: {e}")

    async def _build_surface(
        self,
        symbol: str,
        option_chain: List[OptionQuote],
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float,
    ) -> VolatilitySurface:
        """构建波动率曲面"""
        now = datetime.now()

        # 按到期日分组
        expiry_groups: Dict[datetime, List[OptionQuote]] = {}
        for quote in option_chain:
            if quote.implied_vol is not None and quote.implied_vol > 0:
                if quote.expiry not in expiry_groups:
                    expiry_groups[quote.expiry] = []
                expiry_groups[quote.expiry].append(quote)

        # 构建切片
        slices = []
        for expiry, quotes in sorted(expiry_groups.items()):
            time_to_expiry = (expiry - now).total_seconds() / (365.25 * 24 * 3600)

            if time_to_expiry < self._config.min_time_to_expiry:
                continue
            if time_to_expiry > self._config.max_time_to_expiry:
                continue
            if len(quotes) < self._config.min_points_per_slice:
                continue

            # 计算远期价格
            forward_price = spot_price * math.exp(
                (risk_free_rate - dividend_yield) * time_to_expiry
            )

            slice_ = self._build_slice(
                expiry, time_to_expiry, quotes, spot_price, forward_price
            )
            if slice_:
                slices.append(slice_)

        # 排序切片
        slices.sort(key=lambda s: s.time_to_expiry)

        # 计算期限结构
        atm_term_structure = [
            (s.time_to_expiry, s.atm_vol) for s in slices
        ]
        term_structure_shape = self._classify_term_structure(atm_term_structure)

        # 计算整体指标
        overall_level = sum(s.atm_vol for s in slices) / len(slices) if slices else 0
        average_skew = sum(s.skew for s in slices) / len(slices) if slices else 0
        average_butterfly = sum(s.butterfly for s in slices) / len(slices) if slices else 0

        return VolatilitySurface(
            symbol=symbol,
            spot_price=spot_price,
            timestamp=now,
            slices=slices,
            term_structure_shape=term_structure_shape,
            atm_term_structure=atm_term_structure,
            overall_level=overall_level,
            average_skew=average_skew,
            average_butterfly=average_butterfly,
            interpolation_method=self._config.interpolation_method,
            state=SurfaceState.READY,
        )

    def _build_slice(
        self,
        expiry: datetime,
        time_to_expiry: float,
        quotes: List[OptionQuote],
        spot_price: float,
        forward_price: float,
    ) -> Optional[VolatilitySlice]:
        """构建波动率切片"""
        # 转换为波动率点
        points = []
        for quote in quotes:
            if quote.implied_vol is None or quote.implied_vol <= 0:
                continue

            moneyness = quote.strike / spot_price
            log_moneyness = math.log(quote.strike / forward_price)

            point = VolatilityPoint(
                strike=quote.strike,
                expiry=expiry,
                implied_vol=quote.implied_vol,
                moneyness=moneyness,
                time_to_expiry=time_to_expiry,
                log_moneyness=log_moneyness,
                forward_price=forward_price,
                option_type=quote.option_type,
                bid_vol=quote.implied_vol * 0.98 if quote.bid > 0 else None,
                ask_vol=quote.implied_vol * 1.02 if quote.ask > 0 else None,
            )
            points.append(point)

        if not points:
            return None

        # 找到ATM点
        atm_strike = min(points, key=lambda p: abs(p.moneyness - 1.0)).strike
        atm_vol = min(points, key=lambda p: abs(p.moneyness - 1.0)).implied_vol

        # 计算偏度和蝶式
        skew, butterfly = self._calculate_skew_butterfly(points)

        # 分类微笑形态
        smile_shape = self._classify_smile(points, atm_vol)

        # 计算Delta波动率
        put_25d_vol = self._find_delta_vol(points, -0.25, OptionType.PUT)
        call_25d_vol = self._find_delta_vol(points, 0.25, OptionType.CALL)
        put_10d_vol = self._find_delta_vol(points, -0.10, OptionType.PUT)
        call_10d_vol = self._find_delta_vol(points, 0.10, OptionType.CALL)

        return VolatilitySlice(
            expiry=expiry,
            time_to_expiry=time_to_expiry,
            atm_vol=atm_vol,
            points=points,
            smile_shape=smile_shape,
            skew=skew,
            butterfly=butterfly,
            atm_strike=atm_strike,
            forward_price=forward_price,
            put_25d_vol=put_25d_vol,
            call_25d_vol=call_25d_vol,
            put_10d_vol=put_10d_vol,
            call_10d_vol=call_10d_vol,
        )

    def _calculate_skew_butterfly(
        self, points: List[VolatilityPoint]
    ) -> Tuple[float, float]:
        """计算偏度和蝶式"""
        # 找到ATM、25D Put、25D Call的波动率
        atm_point = min(points, key=lambda p: abs(p.moneyness - 1.0))

        # 近似25D点（货币性约0.95和1.05）
        put_points = [p for p in points if p.moneyness < 1.0]
        call_points = [p for p in points if p.moneyness > 1.0]

        if not put_points or not call_points:
            return 0.0, 0.0

        put_25d = min(put_points, key=lambda p: abs(p.moneyness - 0.95))
        call_25d = min(call_points, key=lambda p: abs(p.moneyness - 1.05))

        # 25D Risk Reversal (Skew)
        skew = call_25d.implied_vol - put_25d.implied_vol

        # 25D Butterfly
        butterfly = (put_25d.implied_vol + call_25d.implied_vol) / 2 - atm_point.implied_vol

        return skew, butterfly

    def _classify_smile(
        self, points: List[VolatilityPoint], atm_vol: float
    ) -> SmileShape:
        """分类微笑形态"""
        if len(points) < 3:
            return SmileShape.FLAT

        put_points = [p for p in points if p.moneyness < 0.98]
        call_points = [p for p in points if p.moneyness > 1.02]

        if not put_points or not call_points:
            return SmileShape.FLAT

        avg_put_vol = sum(p.implied_vol for p in put_points) / len(put_points)
        avg_call_vol = sum(p.implied_vol for p in call_points) / len(call_points)

        put_premium = avg_put_vol - atm_vol
        call_premium = avg_call_vol - atm_vol

        # 分类
        threshold = 0.01

        if put_premium > threshold and call_premium > threshold:
            return SmileShape.SMILE
        elif put_premium < -threshold and call_premium < -threshold:
            return SmileShape.FROWN
        elif put_premium > threshold and call_premium < -threshold:
            return SmileShape.SKEW_LEFT
        elif put_premium < -threshold and call_premium > threshold:
            return SmileShape.SKEW_RIGHT
        elif abs(put_premium - call_premium) > threshold:
            return SmileShape.SMIRK
        else:
            return SmileShape.FLAT

    def _classify_term_structure(
        self, atm_term_structure: List[Tuple[float, float]]
    ) -> TermStructureShape:
        """分类期限结构形态"""
        if len(atm_term_structure) < 2:
            return TermStructureShape.FLAT

        vols = [v for _, v in atm_term_structure]
        times = [t for t, _ in atm_term_structure]

        # 计算斜率
        if len(vols) >= 2:
            slope = (vols[-1] - vols[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0
        else:
            slope = 0

        # 检查是否有驼峰
        has_hump = False
        if len(vols) >= 3:
            for i in range(1, len(vols) - 1):
                if vols[i] > vols[i-1] and vols[i] > vols[i+1]:
                    has_hump = True
                    break

        # 检查是否U形
        is_u_shaped = False
        if len(vols) >= 3:
            min_idx = vols.index(min(vols))
            if 0 < min_idx < len(vols) - 1:
                is_u_shaped = True

        # 分类
        threshold = 0.02

        if has_hump:
            return TermStructureShape.HUMPED
        elif is_u_shaped:
            return TermStructureShape.U_SHAPED
        elif slope > threshold:
            return TermStructureShape.NORMAL
        elif slope < -threshold:
            return TermStructureShape.INVERTED
        else:
            return TermStructureShape.FLAT

    def _find_delta_vol(
        self,
        points: List[VolatilityPoint],
        target_delta: float,
        option_type: OptionType,
    ) -> Optional[float]:
        """查找指定Delta的波动率"""
        # 简化实现：根据货币性近似
        if option_type == OptionType.PUT:
            target_moneyness = 1.0 + target_delta  # -0.25 -> 0.75
        else:
            target_moneyness = 1.0 + target_delta  # 0.25 -> 1.25

        filtered = [p for p in points if p.option_type == option_type]
        if not filtered:
            filtered = points

        if not filtered:
            return None

        closest = min(filtered, key=lambda p: abs(p.moneyness - target_moneyness))
        return closest.implied_vol

    # ============ Change Detection ============

    def _detect_change(
        self,
        symbol: str,
        old_surface: VolatilitySurface,
        new_surface: VolatilitySurface,
    ) -> Optional[SurfaceChange]:
        """检测曲面变化"""
        level_change = new_surface.overall_level - old_surface.overall_level
        level_change_pct = level_change / old_surface.overall_level if old_surface.overall_level > 0 else 0

        skew_change = new_surface.average_skew - old_surface.average_skew
        butterfly_change = new_surface.average_butterfly - old_surface.average_butterfly

        # 计算期限结构变化
        term_structure_change = 0.0
        if old_surface.atm_term_structure and new_surface.atm_term_structure:
            old_slope = self._calculate_term_slope(old_surface.atm_term_structure)
            new_slope = self._calculate_term_slope(new_surface.atm_term_structure)
            term_structure_change = new_slope - old_slope

        # 找到最大变化点
        max_change = 0.0
        max_strike = 0.0
        max_expiry = datetime.now()

        for new_slice in new_surface.slices:
            for old_slice in old_surface.slices:
                if new_slice.expiry == old_slice.expiry:
                    for new_point in new_slice.points:
                        for old_point in old_slice.points:
                            if new_point.strike == old_point.strike:
                                change = abs(new_point.implied_vol - old_point.implied_vol)
                                if change > max_change:
                                    max_change = change
                                    max_strike = new_point.strike
                                    max_expiry = new_point.expiry

        return SurfaceChange(
            symbol=symbol,
            timestamp=datetime.now(),
            previous_level=old_surface.overall_level,
            current_level=new_surface.overall_level,
            level_change=level_change,
            level_change_pct=level_change_pct,
            skew_change=skew_change,
            butterfly_change=butterfly_change,
            term_structure_change=term_structure_change,
            smile_shape_changed=any(
                ns.smile_shape != os.smile_shape
                for ns in new_surface.slices
                for os in old_surface.slices
                if ns.expiry == os.expiry
            ),
            term_structure_shape_changed=(
                new_surface.term_structure_shape != old_surface.term_structure_shape
            ),
            max_point_change=max_change,
            max_point_strike=max_strike,
            max_point_expiry=max_expiry,
        )

    def _calculate_term_slope(
        self, term_structure: List[Tuple[float, float]]
    ) -> float:
        """计算期限结构斜率"""
        if len(term_structure) < 2:
            return 0.0

        times = [t for t, _ in term_structure]
        vols = [v for _, v in term_structure]

        return (vols[-1] - vols[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0

    # ============ Alert Detection ============

    def _check_alerts(
        self,
        symbol: str,
        change: SurfaceChange,
        surface: VolatilitySurface,
    ) -> List[VolatilityAlert]:
        """检查警报"""
        alerts = []

        # 检查水平变化
        if abs(change.level_change_pct) > self._config.level_change_alert_threshold:
            severity = "high" if abs(change.level_change_pct) > 0.05 else "medium"
            alerts.append(VolatilityAlert(
                symbol=symbol,
                alert_type=AlertType.SURFACE_SHIFT,
                severity=severity,
                message=f"Volatility surface shifted {change.level_change_pct:.2%}",
                details={
                    "previous_level": change.previous_level,
                    "current_level": change.current_level,
                    "change_pct": change.level_change_pct,
                },
            ))

        # 检查偏度变化
        if abs(change.skew_change) > self._config.skew_change_alert_threshold:
            alerts.append(VolatilityAlert(
                symbol=symbol,
                alert_type=AlertType.SMILE_CHANGE,
                severity="medium",
                message=f"Volatility skew changed by {change.skew_change:.4f}",
                details={
                    "skew_change": change.skew_change,
                    "current_skew": surface.average_skew,
                },
            ))

        # 检查期限结构变化
        if change.term_structure_shape_changed:
            alerts.append(VolatilityAlert(
                symbol=symbol,
                alert_type=AlertType.TERM_STRUCTURE_CHANGE,
                severity="medium",
                message="Term structure shape changed",
                details={
                    "new_shape": surface.term_structure_shape.value,
                },
            ))

        # 检查极端波动率
        if surface.overall_level < self._config.extreme_vol_low:
            alerts.append(VolatilityAlert(
                symbol=symbol,
                alert_type=AlertType.EXTREME_VOL,
                severity="high",
                message=f"Extremely low volatility: {surface.overall_level:.2%}",
                details={"level": surface.overall_level},
            ))
        elif surface.overall_level > self._config.extreme_vol_high:
            alerts.append(VolatilityAlert(
                symbol=symbol,
                alert_type=AlertType.EXTREME_VOL,
                severity="critical",
                message=f"Extremely high volatility: {surface.overall_level:.2%}",
                details={"level": surface.overall_level},
            ))

        return alerts

    # ============ Arbitrage Detection ============

    async def _detect_arbitrage(
        self,
        symbol: str,
        surface: VolatilitySurface,
    ) -> List[ArbitrageOpportunity]:
        """检测套利机会"""
        opportunities = []

        # 日历套利检测
        calendar_arb = self._detect_calendar_arbitrage(surface)
        opportunities.extend(calendar_arb)

        # 蝶式套利检测
        butterfly_arb = self._detect_butterfly_arbitrage(surface)
        opportunities.extend(butterfly_arb)

        return opportunities

    def _detect_calendar_arbitrage(
        self, surface: VolatilitySurface
    ) -> List[ArbitrageOpportunity]:
        """检测日历套利"""
        opportunities = []

        if len(surface.slices) < 2:
            return opportunities

        for i in range(len(surface.slices) - 1):
            near_slice = surface.slices[i]
            far_slice = surface.slices[i + 1]

            # 检查相同行权价的波动率是否倒挂（近期高于远期过多）
            for near_point in near_slice.points:
                for far_point in far_slice.points:
                    if abs(near_point.strike - far_point.strike) < 0.01:
                        # 日历套利：近期波动率显著高于远期
                        if near_point.implied_vol > far_point.implied_vol + self._config.arbitrage_min_profit:
                            opportunities.append(ArbitrageOpportunity(
                                symbol=surface.symbol,
                                arbitrage_type="calendar",
                                description=f"Calendar spread arbitrage at strike {near_point.strike}",
                                strikes=[near_point.strike],
                                expiries=[near_slice.expiry, far_slice.expiry],
                                implied_profit=near_point.implied_vol - far_point.implied_vol,
                                bid_ask_cost=0.002,  # 估计买卖价差成本
                                net_profit=near_point.implied_vol - far_point.implied_vol - 0.002,
                                confidence=0.7,
                            ))

        return opportunities

    def _detect_butterfly_arbitrage(
        self, surface: VolatilitySurface
    ) -> List[ArbitrageOpportunity]:
        """检测蝶式套利"""
        opportunities = []

        for slice_ in surface.slices:
            if len(slice_.points) < 3:
                continue

            sorted_points = sorted(slice_.points, key=lambda p: p.strike)

            # 检查蝶式不等式违反
            for i in range(1, len(sorted_points) - 1):
                lower = sorted_points[i - 1]
                middle = sorted_points[i]
                upper = sorted_points[i + 1]

                # 蝶式条件：中间行权价的波动率应该低于两端的加权平均
                k1, k2, k3 = lower.strike, middle.strike, upper.strike
                v1, v2, v3 = lower.implied_vol, middle.implied_vol, upper.implied_vol

                # 线性插值的波动率
                weight = (k2 - k1) / (k3 - k1) if k3 != k1 else 0.5
                interpolated_vol = v1 * (1 - weight) + v3 * weight

                # 如果实际波动率显著高于插值，可能存在套利
                if v2 > interpolated_vol + self._config.arbitrage_min_profit:
                    opportunities.append(ArbitrageOpportunity(
                        symbol=surface.symbol,
                        arbitrage_type="butterfly",
                        description=f"Butterfly arbitrage at strikes {k1}/{k2}/{k3}",
                        strikes=[k1, k2, k3],
                        expiries=[slice_.expiry],
                        implied_profit=v2 - interpolated_vol,
                        bid_ask_cost=0.003,
                        net_profit=v2 - interpolated_vol - 0.003,
                        confidence=0.6,
                    ))

        return opportunities

    # ============ Historical Vol Comparison ============

    async def get_vol_comparison(
        self, symbol: str
    ) -> Optional[HistoricalVolComparison]:
        """获取历史波动率对比"""
        surface = self._surfaces.get(symbol)
        if not surface or not surface.slices:
            return None

        # 获取ATM隐含波动率
        implied_vol = surface.overall_level

        # 获取实现波动率
        rv_1d = await self._historical_provider.get_realized_volatility(symbol, 1)
        rv_5d = await self._historical_provider.get_realized_volatility(symbol, 5)
        rv_20d = await self._historical_provider.get_realized_volatility(symbol, 20)
        rv_60d = await self._historical_provider.get_realized_volatility(symbol, 60)

        # 计算波动率溢价
        vol_premium = implied_vol - rv_20d

        # 计算溢价百分位（简化实现）
        vol_premium_percentile = 50.0  # 需要历史数据计算

        return HistoricalVolComparison(
            symbol=symbol,
            timestamp=datetime.now(),
            implied_vol=implied_vol,
            realized_vol_1d=rv_1d,
            realized_vol_5d=rv_5d,
            realized_vol_20d=rv_20d,
            realized_vol_60d=rv_60d,
            vol_premium=vol_premium,
            vol_premium_percentile=vol_premium_percentile,
        )

    # ============ History Management ============

    def _add_to_history(self, symbol: str, surface: VolatilitySurface) -> None:
        """添加到历史记录"""
        if symbol not in self._surface_history:
            self._surface_history[symbol] = []

        self._surface_history[symbol].append(surface)

        # 限制大小
        if len(self._surface_history[symbol]) > self._config.history_size:
            self._surface_history[symbol] = self._surface_history[symbol][-self._config.history_size:]

    # ============ Callbacks ============

    def on_surface_update(
        self, callback: Callable[[VolatilitySurface], None]
    ) -> None:
        """注册曲面更新回调"""
        self._surface_callbacks.append(callback)

    def on_change(
        self, callback: Callable[[SurfaceChange], None]
    ) -> None:
        """注册变化回调"""
        self._change_callbacks.append(callback)

    def on_alert(
        self, callback: Callable[[VolatilityAlert], None]
    ) -> None:
        """注册警报回调"""
        self._alert_callbacks.append(callback)

    def on_arbitrage(
        self, callback: Callable[[ArbitrageOpportunity], None]
    ) -> None:
        """注册套利机会回调"""
        self._arbitrage_callbacks.append(callback)

    def _notify_surface(self, surface: VolatilitySurface) -> None:
        """通知曲面更新"""
        for callback in self._surface_callbacks:
            try:
                callback(surface)
            except Exception as e:
                logger.error(f"Error in surface callback: {e}")

    def _notify_change(self, change: SurfaceChange) -> None:
        """通知变化"""
        for callback in self._change_callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    def _notify_alert(self, alert: VolatilityAlert) -> None:
        """通知警报"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _notify_arbitrage(self, opportunity: ArbitrageOpportunity) -> None:
        """通知套利机会"""
        for callback in self._arbitrage_callbacks:
            try:
                callback(opportunity)
            except Exception as e:
                logger.error(f"Error in arbitrage callback: {e}")

    # ============ Public API ============

    def get_surface(self, symbol: str) -> Optional[VolatilitySurface]:
        """获取波动率曲面"""
        return self._surfaces.get(symbol)

    def get_volatility(
        self, symbol: str, strike: float, time_to_expiry: float
    ) -> Optional[float]:
        """获取指定点的波动率"""
        surface = self._surfaces.get(symbol)
        if surface:
            return surface.get_vol(strike, time_to_expiry)
        return None

    def get_atm_volatility(
        self, symbol: str, time_to_expiry: Optional[float] = None
    ) -> Optional[float]:
        """获取ATM波动率"""
        surface = self._surfaces.get(symbol)
        if not surface or not surface.slices:
            return None

        if time_to_expiry is None:
            # 返回最近到期的ATM波动率
            return surface.slices[0].atm_vol

        # 插值
        return surface.get_vol(surface.spot_price, time_to_expiry)

    def get_smile(
        self, symbol: str, time_to_expiry: float
    ) -> Optional[VolatilitySlice]:
        """获取指定到期时间的微笑"""
        surface = self._surfaces.get(symbol)
        if not surface:
            return None

        # 找到最近的切片
        closest = min(
            surface.slices,
            key=lambda s: abs(s.time_to_expiry - time_to_expiry),
            default=None,
        )
        return closest

    def get_term_structure(
        self, symbol: str
    ) -> List[Tuple[float, float]]:
        """获取期限结构"""
        surface = self._surfaces.get(symbol)
        if surface:
            return surface.atm_term_structure
        return []

    def get_skew(self, symbol: str) -> Optional[float]:
        """获取平均偏度"""
        surface = self._surfaces.get(symbol)
        if surface:
            return surface.average_skew
        return None

    def get_butterfly(self, symbol: str) -> Optional[float]:
        """获取平均蝶式"""
        surface = self._surfaces.get(symbol)
        if surface:
            return surface.average_butterfly
        return None

    def get_surface_history(
        self, symbol: str, limit: int = 100
    ) -> List[VolatilitySurface]:
        """获取曲面历史"""
        history = self._surface_history.get(symbol, [])
        return history[-limit:]

    def get_change_history(
        self, symbol: str, limit: int = 100
    ) -> List[SurfaceChange]:
        """获取变化历史"""
        history = self._change_history.get(symbol, [])
        return history[-limit:]

    def get_alerts(self, limit: int = 100) -> List[VolatilityAlert]:
        """获取警报列表"""
        return self._alert_history[-limit:]

    def get_arbitrage_opportunities(
        self, limit: int = 100
    ) -> List[ArbitrageOpportunity]:
        """获取套利机会列表"""
        return self._arbitrage_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "symbols_tracked": len(self._symbols),
            "surfaces_built": len(self._surfaces),
            "total_alerts": len(self._alert_history),
            "total_arbitrage_opportunities": len(self._arbitrage_history),
            "surfaces": {
                symbol: {
                    "state": surface.state.value,
                    "slices": len(surface.slices),
                    "overall_level": surface.overall_level,
                    "term_structure": surface.term_structure_shape.value,
                }
                for symbol, surface in self._surfaces.items()
            },
        }


# ============ Factory Functions ============


def create_surface_manager(
    interpolation_method: str = "linear",
    min_points_per_slice: int = 3,
    level_change_alert_threshold: float = 0.02,
    option_provider: Optional[OptionDataProvider] = None,
    historical_provider: Optional[HistoricalVolProvider] = None,
) -> VolatilitySurfaceManager:
    """创建波动率曲面管理器"""
    config = SurfaceManagerConfig(
        interpolation_method=InterpolationMethod(interpolation_method),
        min_points_per_slice=min_points_per_slice,
        level_change_alert_threshold=level_change_alert_threshold,
    )
    return VolatilitySurfaceManager(
        config=config,
        option_provider=option_provider,
        historical_provider=historical_provider,
    )


def create_option_quote(
    symbol: str,
    option_symbol: str,
    option_type: str,
    strike: float,
    expiry: datetime,
    bid: float,
    ask: float,
    implied_vol: float,
    volume: int = 0,
    open_interest: int = 0,
) -> OptionQuote:
    """创建期权报价"""
    mid = (bid + ask) / 2
    return OptionQuote(
        symbol=symbol,
        option_symbol=option_symbol,
        option_type=OptionType(option_type),
        strike=strike,
        expiry=expiry,
        bid=bid,
        ask=ask,
        mid=mid,
        last=mid,
        volume=volume,
        open_interest=open_interest,
        implied_vol=implied_vol,
    )


async def quick_vol_lookup(
    manager: VolatilitySurfaceManager,
    symbol: str,
    strike: float,
    time_to_expiry: float,
) -> Optional[float]:
    """快速查询波动率"""
    return manager.get_volatility(symbol, strike, time_to_expiry)


# ============ Exports ============


__all__ = [
    # Enums
    "OptionType",
    "InterpolationMethod",
    "SmileShape",
    "TermStructureShape",
    "AlertType",
    "SurfaceState",
    # Data Classes
    "OptionQuote",
    "VolatilityPoint",
    "VolatilitySlice",
    "VolatilitySurface",
    "SurfaceChange",
    "ArbitrageOpportunity",
    "VolatilityAlert",
    "HistoricalVolComparison",
    "SurfaceManagerConfig",
    # Interfaces
    "OptionDataProvider",
    "HistoricalVolProvider",
    # Simulated Providers
    "SimulatedOptionDataProvider",
    "SimulatedHistoricalVolProvider",
    # Main Class
    "VolatilitySurfaceManager",
    # Factory Functions
    "create_surface_manager",
    "create_option_quote",
    "quick_vol_lookup",
]
