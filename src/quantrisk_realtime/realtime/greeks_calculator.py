"""
Greeks Calculator - 希腊字母计算器

实时期权希腊字母计算和风险敞口监控系统。
支持Black-Scholes、Binomial Tree等定价模型。

核心功能：
1. 单期权希腊字母计算 - Delta, Gamma, Theta, Vega, Rho
2. 高阶希腊字母 - Vanna, Volga, Charm, Veta等
3. 组合希腊字母汇总 - 组合级别风险敞口
4. 希腊字母监控 - 实时监控和警报
5. 情景分析 - What-if分析
6. P&L归因 - 基于希腊字母的损益分解
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Constants ============


# 标准正态分布相关常量
SQRT_2PI = math.sqrt(2 * math.pi)


# ============ Enums ============


class OptionType(str, Enum):
    """期权类型"""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """期权风格"""
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingModel(str, Enum):
    """定价模型"""
    BLACK_SCHOLES = "black_scholes"
    BLACK_76 = "black_76"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


class GreekType(str, Enum):
    """希腊字母类型"""
    DELTA = "delta"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"
    RHO = "rho"
    # 高阶希腊字母
    VANNA = "vanna"       # d(Delta)/d(Vol)
    VOLGA = "volga"       # d(Vega)/d(Vol)
    CHARM = "charm"       # d(Delta)/d(Time)
    VETA = "veta"         # d(Vega)/d(Time)
    SPEED = "speed"       # d(Gamma)/d(Spot)
    ZOMMA = "zomma"       # d(Gamma)/d(Vol)
    COLOR = "color"       # d(Gamma)/d(Time)


class AlertSeverity(str, Enum):
    """警报严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CalculatorState(str, Enum):
    """计算器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Math Utilities ============


def norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数"""
    return math.exp(-0.5 * x * x) / SQRT_2PI


# ============ Data Classes ============


@dataclass
class OptionPosition:
    """期权持仓"""
    position_id: str
    symbol: str                    # 标的代码
    option_symbol: str             # 期权代码
    option_type: OptionType        # 期权类型
    option_style: OptionStyle      # 期权风格
    strike: float                  # 行权价
    expiry: datetime               # 到期日
    quantity: int                  # 持仓数量（正为多头，负为空头）
    multiplier: float = 100.0      # 合约乘数
    entry_price: float = 0.0       # 建仓价格
    current_price: float = 0.0     # 当前价格

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "option_type": self.option_type.value,
            "option_style": self.option_style.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
        }


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    spot_price: float              # 现货价格
    volatility: float              # 波动率
    risk_free_rate: float          # 无风险利率
    dividend_yield: float = 0.0    # 股息率
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "volatility": self.volatility,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Greeks:
    """希腊字母"""
    delta: float = 0.0             # Delta
    gamma: float = 0.0             # Gamma
    theta: float = 0.0             # Theta (每日)
    vega: float = 0.0              # Vega (1%波动率变化)
    rho: float = 0.0               # Rho (1%利率变化)
    # 高阶希腊字母
    vanna: float = 0.0             # Vanna
    volga: float = 0.0             # Volga (Vomma)
    charm: float = 0.0             # Charm
    veta: float = 0.0              # Veta
    speed: float = 0.0             # Speed
    zomma: float = 0.0             # Zomma
    color: float = 0.0             # Color

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "veta": self.veta,
            "speed": self.speed,
            "zomma": self.zomma,
            "color": self.color,
        }

    def scale(self, factor: float) -> "Greeks":
        """按因子缩放"""
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
            vanna=self.vanna * factor,
            volga=self.volga * factor,
            charm=self.charm * factor,
            veta=self.veta * factor,
            speed=self.speed * factor,
            zomma=self.zomma * factor,
            color=self.color * factor,
        )

    def add(self, other: "Greeks") -> "Greeks":
        """加法"""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            charm=self.charm + other.charm,
            veta=self.veta + other.veta,
            speed=self.speed + other.speed,
            zomma=self.zomma + other.zomma,
            color=self.color + other.color,
        )


@dataclass
class OptionPricing:
    """期权定价结果"""
    theoretical_price: float       # 理论价格
    intrinsic_value: float         # 内在价值
    time_value: float              # 时间价值
    greeks: Greeks                 # 希腊字母
    model: PricingModel            # 使用的模型
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theoretical_price": self.theoretical_price,
            "intrinsic_value": self.intrinsic_value,
            "time_value": self.time_value,
            "greeks": self.greeks.to_dict(),
            "model": self.model.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PositionGreeks:
    """持仓希腊字母"""
    position_id: str
    option_symbol: str
    quantity: int
    multiplier: float
    unit_greeks: Greeks            # 单位希腊字母
    position_greeks: Greeks        # 持仓希腊字母（乘以数量和乘数）
    notional_value: float          # 名义价值
    market_value: float            # 市场价值
    delta_equivalent: float        # Delta等价股票数量
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "option_symbol": self.option_symbol,
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "unit_greeks": self.unit_greeks.to_dict(),
            "position_greeks": self.position_greeks.to_dict(),
            "notional_value": self.notional_value,
            "market_value": self.market_value,
            "delta_equivalent": self.delta_equivalent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioGreeks:
    """组合希腊字母"""
    portfolio_id: str
    timestamp: datetime
    positions: List[PositionGreeks]
    total_greeks: Greeks
    total_delta_equivalent: float
    total_notional: float
    total_market_value: float
    # 按标的汇总
    by_underlying: Dict[str, Greeks]
    # 按到期日汇总
    by_expiry: Dict[str, Greeks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp.isoformat(),
            "positions": [p.to_dict() for p in self.positions],
            "total_greeks": self.total_greeks.to_dict(),
            "total_delta_equivalent": self.total_delta_equivalent,
            "total_notional": self.total_notional,
            "total_market_value": self.total_market_value,
            "by_underlying": {k: v.to_dict() for k, v in self.by_underlying.items()},
            "by_expiry": {k: v.to_dict() for k, v in self.by_expiry.items()},
        }


@dataclass
class ScenarioResult:
    """情景分析结果"""
    scenario_name: str
    spot_change: float             # 现货变化百分比
    vol_change: float              # 波动率变化（绝对值）
    time_change: float             # 时间变化（天）
    rate_change: float             # 利率变化（绝对值）
    new_price: float               # 新理论价格
    price_change: float            # 价格变化
    pnl: float                     # 预计损益
    new_greeks: Greeks             # 新希腊字母
    greeks_change: Greeks          # 希腊字母变化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "spot_change": self.spot_change,
            "vol_change": self.vol_change,
            "time_change": self.time_change,
            "rate_change": self.rate_change,
            "new_price": self.new_price,
            "price_change": self.price_change,
            "pnl": self.pnl,
            "new_greeks": self.new_greeks.to_dict(),
            "greeks_change": self.greeks_change.to_dict(),
        }


@dataclass
class PnLAttribution:
    """P&L归因"""
    position_id: str
    total_pnl: float
    delta_pnl: float               # Delta贡献
    gamma_pnl: float               # Gamma贡献
    theta_pnl: float               # Theta贡献
    vega_pnl: float                # Vega贡献
    rho_pnl: float                 # Rho贡献
    unexplained_pnl: float         # 未解释部分
    spot_change: float             # 标的变化
    vol_change: float              # 波动率变化
    time_passed: float             # 经过时间（天）
    rate_change: float             # 利率变化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "total_pnl": self.total_pnl,
            "delta_pnl": self.delta_pnl,
            "gamma_pnl": self.gamma_pnl,
            "theta_pnl": self.theta_pnl,
            "vega_pnl": self.vega_pnl,
            "rho_pnl": self.rho_pnl,
            "unexplained_pnl": self.unexplained_pnl,
            "spot_change": self.spot_change,
            "vol_change": self.vol_change,
            "time_passed": self.time_passed,
            "rate_change": self.rate_change,
        }


@dataclass
class GreeksAlert:
    """希腊字母警报"""
    alert_id: str
    position_id: Optional[str]
    greek_type: GreekType
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "position_id": self.position_id,
            "greek_type": self.greek_type.value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GreeksLimit:
    """希腊字母限额"""
    greek_type: GreekType
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    warning_threshold: float = 0.8  # 预警阈值（限额的百分比）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "greek_type": self.greek_type.value,
            "max_value": self.max_value,
            "min_value": self.min_value,
            "warning_threshold": self.warning_threshold,
        }


@dataclass
class CalculatorConfig:
    """计算器配置"""
    pricing_model: PricingModel = PricingModel.BLACK_SCHOLES
    calculate_higher_order: bool = True   # 是否计算高阶希腊字母
    update_interval_ms: int = 1000        # 更新间隔
    history_size: int = 1000              # 历史记录大小
    # 默认限额
    delta_limit: float = 1000.0           # Delta限额
    gamma_limit: float = 100.0            # Gamma限额
    vega_limit: float = 10000.0           # Vega限额
    theta_limit: float = 5000.0           # Theta限额


# ============ Abstract Interfaces ============


class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取市场数据"""
        pass

    @abstractmethod
    async def get_spot_price(self, symbol: str) -> float:
        """获取现货价格"""
        pass


class PositionProvider(ABC):
    """持仓提供者接口"""

    @abstractmethod
    async def get_positions(self) -> List[OptionPosition]:
        """获取所有期权持仓"""
        pass

    @abstractmethod
    async def get_position(self, position_id: str) -> Optional[OptionPosition]:
        """获取单个持仓"""
        pass


# ============ Simulated Providers ============


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self._market_data: Dict[str, MarketData] = {}

    def set_market_data(self, symbol: str, data: MarketData) -> None:
        """设置市场数据"""
        self._market_data[symbol] = data

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        return self._market_data.get(symbol)

    async def get_spot_price(self, symbol: str) -> float:
        data = self._market_data.get(symbol)
        return data.spot_price if data else 100.0


class SimulatedPositionProvider(PositionProvider):
    """模拟持仓提供者"""

    def __init__(self):
        self._positions: Dict[str, OptionPosition] = {}

    def add_position(self, position: OptionPosition) -> None:
        """添加持仓"""
        self._positions[position.position_id] = position

    def remove_position(self, position_id: str) -> None:
        """移除持仓"""
        self._positions.pop(position_id, None)

    async def get_positions(self) -> List[OptionPosition]:
        return list(self._positions.values())

    async def get_position(self, position_id: str) -> Optional[OptionPosition]:
        return self._positions.get(position_id)


# ============ Pricing Models ============


class BlackScholesPricer:
    """Black-Scholes定价器"""

    @staticmethod
    def calculate_d1_d2(
        S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> Tuple[float, float]:
        """计算d1和d2"""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @staticmethod
    def price(
        option_type: OptionType,
        S: float,      # 现货价格
        K: float,      # 行权价
        T: float,      # 到期时间（年）
        r: float,      # 无风险利率
        q: float,      # 股息率
        sigma: float,  # 波动率
    ) -> float:
        """计算期权价格"""
        if T <= 0:
            # 到期时
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1, d2 = BlackScholesPricer.calculate_d1_d2(S, K, T, r, q, sigma)

        if option_type == OptionType.CALL:
            price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

        return max(price, 0)

    @staticmethod
    def greeks(
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        calculate_higher_order: bool = True,
    ) -> Greeks:
        """计算希腊字母"""
        if T <= 0 or sigma <= 0:
            # 到期或无效波动率
            delta = 1.0 if option_type == OptionType.CALL and S > K else 0.0
            if option_type == OptionType.PUT:
                delta = -1.0 if S < K else 0.0
            return Greeks(delta=delta)

        d1, d2 = BlackScholesPricer.calculate_d1_d2(S, K, T, r, q, sigma)
        sqrt_T = math.sqrt(T)

        # 基本希腊字母
        nd1 = norm_pdf(d1)
        Nd1 = norm_cdf(d1)
        Nd2 = norm_cdf(d2)

        # Delta
        if option_type == OptionType.CALL:
            delta = math.exp(-q * T) * Nd1
        else:
            delta = math.exp(-q * T) * (Nd1 - 1)

        # Gamma (对Call和Put相同)
        gamma = math.exp(-q * T) * nd1 / (S * sigma * sqrt_T)

        # Theta (每日)
        term1 = -S * sigma * math.exp(-q * T) * nd1 / (2 * sqrt_T)
        if option_type == OptionType.CALL:
            term2 = -r * K * math.exp(-r * T) * Nd2
            term3 = q * S * math.exp(-q * T) * Nd1
        else:
            term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
            term3 = -q * S * math.exp(-q * T) * norm_cdf(-d1)
        theta = (term1 + term2 + term3) / 365  # 转换为每日

        # Vega (1%波动率变化)
        vega = S * math.exp(-q * T) * sqrt_T * nd1 / 100

        # Rho (1%利率变化)
        if option_type == OptionType.CALL:
            rho = K * T * math.exp(-r * T) * Nd2 / 100
        else:
            rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100

        result = Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

        # 高阶希腊字母
        if calculate_higher_order:
            # Vanna: d(Delta)/d(Vol) = d(Vega)/d(S)
            vanna = -math.exp(-q * T) * nd1 * d2 / sigma

            # Volga (Vomma): d(Vega)/d(Vol)
            volga = S * math.exp(-q * T) * sqrt_T * nd1 * d1 * d2 / sigma / 100

            # Charm: d(Delta)/d(Time)
            charm_factor = math.exp(-q * T) * nd1
            if option_type == OptionType.CALL:
                charm = q * math.exp(-q * T) * Nd1 - charm_factor * (
                    2 * (r - q) * T - d2 * sigma * sqrt_T
                ) / (2 * T * sigma * sqrt_T)
            else:
                charm = -q * math.exp(-q * T) * norm_cdf(-d1) - charm_factor * (
                    2 * (r - q) * T - d2 * sigma * sqrt_T
                ) / (2 * T * sigma * sqrt_T)

            # Veta: d(Vega)/d(Time)
            veta = S * math.exp(-q * T) * nd1 * sqrt_T * (
                q + (r - q) * d1 / (sigma * sqrt_T) - (1 + d1 * d2) / (2 * T)
            ) / 100

            # Speed: d(Gamma)/d(S)
            speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)

            # Zomma: d(Gamma)/d(Vol)
            zomma = gamma * (d1 * d2 - 1) / sigma

            # Color: d(Gamma)/d(Time)
            color = -gamma / (2 * T) * (
                2 * q * T + 1 + d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
            )

            result.vanna = vanna
            result.volga = volga
            result.charm = charm / 365  # 转换为每日
            result.veta = veta / 365
            result.speed = speed
            result.zomma = zomma
            result.color = color / 365

        return result


# ============ Main Class ============


class GreeksCalculator:
    """希腊字母计算器"""

    def __init__(
        self,
        config: CalculatorConfig,
        market_provider: Optional[MarketDataProvider] = None,
        position_provider: Optional[PositionProvider] = None,
    ):
        self._config = config
        self._market_provider = market_provider or simulated_default(SimulatedMarketDataProvider, "data from GreeksCalculator's simulated market_provider (SimulatedMarketDataProvider)")
        self._position_provider = position_provider or simulated_default(SimulatedPositionProvider, "data from GreeksCalculator's simulated position_provider (SimulatedPositionProvider)")

        # 状态
        self._state = CalculatorState.IDLE
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 缓存
        self._position_greeks: Dict[str, PositionGreeks] = {}
        self._portfolio_greeks: Optional[PortfolioGreeks] = None

        # 限额
        self._limits: Dict[GreekType, GreeksLimit] = {}

        # 历史记录
        self._greeks_history: List[PortfolioGreeks] = []
        self._alert_history: List[GreeksAlert] = []
        self._alert_counter = 0

        # 回调
        self._greeks_callbacks: List[Callable[[PortfolioGreeks], None]] = []
        self._alert_callbacks: List[Callable[[GreeksAlert], None]] = []

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动计算器"""
        if self._running:
            return

        self._running = True
        self._state = CalculatorState.RUNNING
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("GreeksCalculator started")

    async def stop(self) -> None:
        """停止计算器"""
        self._running = False
        self._state = CalculatorState.IDLE
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("GreeksCalculator stopped")

    async def _update_loop(self) -> None:
        """更新循环"""
        while self._running:
            try:
                await self._update_all_greeks()
                await asyncio.sleep(self._config.update_interval_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1)

    # ============ Single Option Pricing ============

    def price_option(
        self,
        option_type: OptionType,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> OptionPricing:
        """计算单个期权价格和希腊字母"""
        price = BlackScholesPricer.price(
            option_type, spot, strike, time_to_expiry,
            risk_free_rate, dividend_yield, volatility
        )

        greeks = BlackScholesPricer.greeks(
            option_type, spot, strike, time_to_expiry,
            risk_free_rate, dividend_yield, volatility,
            self._config.calculate_higher_order
        )

        # 计算内在价值和时间价值
        if option_type == OptionType.CALL:
            intrinsic = max(spot - strike, 0)
        else:
            intrinsic = max(strike - spot, 0)
        time_value = price - intrinsic

        return OptionPricing(
            theoretical_price=price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            greeks=greeks,
            model=self._config.pricing_model,
        )

    async def price_position(
        self, position: OptionPosition
    ) -> PositionGreeks:
        """计算持仓希腊字母"""
        market_data = await self._market_provider.get_market_data(position.symbol)
        if not market_data:
            # 使用默认值
            market_data = MarketData(
                symbol=position.symbol,
                spot_price=100.0,
                volatility=0.2,
                risk_free_rate=0.05,
            )

        # 计算到期时间
        now = datetime.now()
        time_to_expiry = max((position.expiry - now).total_seconds() / (365.25 * 24 * 3600), 0)

        # 计算单位希腊字母
        pricing = self.price_option(
            position.option_type,
            market_data.spot_price,
            position.strike,
            time_to_expiry,
            market_data.volatility,
            market_data.risk_free_rate,
            market_data.dividend_yield,
        )

        # 计算持仓希腊字母
        factor = position.quantity * position.multiplier
        position_greeks = pricing.greeks.scale(factor)

        # 计算价值
        notional = abs(position.quantity) * position.multiplier * market_data.spot_price
        market_value = position.quantity * position.multiplier * pricing.theoretical_price
        delta_equivalent = position_greeks.delta * market_data.spot_price

        return PositionGreeks(
            position_id=position.position_id,
            option_symbol=position.option_symbol,
            quantity=position.quantity,
            multiplier=position.multiplier,
            unit_greeks=pricing.greeks,
            position_greeks=position_greeks,
            notional_value=notional,
            market_value=market_value,
            delta_equivalent=delta_equivalent,
        )

    # ============ Portfolio Greeks ============

    async def _update_all_greeks(self) -> None:
        """更新所有希腊字母"""
        try:
            positions = await self._position_provider.get_positions()
            if not positions:
                return

            position_greeks_list = []
            total_greeks = Greeks()
            total_delta_equivalent = 0.0
            total_notional = 0.0
            total_market_value = 0.0
            by_underlying: Dict[str, Greeks] = {}
            by_expiry: Dict[str, Greeks] = {}

            for position in positions:
                pg = await self.price_position(position)
                position_greeks_list.append(pg)
                self._position_greeks[position.position_id] = pg

                # 汇总
                total_greeks = total_greeks.add(pg.position_greeks)
                total_delta_equivalent += pg.delta_equivalent
                total_notional += pg.notional_value
                total_market_value += pg.market_value

                # 按标的汇总
                if position.symbol not in by_underlying:
                    by_underlying[position.symbol] = Greeks()
                by_underlying[position.symbol] = by_underlying[position.symbol].add(pg.position_greeks)

                # 按到期日汇总
                expiry_key = position.expiry.strftime("%Y-%m-%d")
                if expiry_key not in by_expiry:
                    by_expiry[expiry_key] = Greeks()
                by_expiry[expiry_key] = by_expiry[expiry_key].add(pg.position_greeks)

            self._portfolio_greeks = PortfolioGreeks(
                portfolio_id="default",
                timestamp=datetime.now(),
                positions=position_greeks_list,
                total_greeks=total_greeks,
                total_delta_equivalent=total_delta_equivalent,
                total_notional=total_notional,
                total_market_value=total_market_value,
                by_underlying=by_underlying,
                by_expiry=by_expiry,
            )

            # 记录历史
            self._add_to_history(self._portfolio_greeks)

            # 检查限额
            self._check_limits(self._portfolio_greeks)

            # 通知回调
            self._notify_greeks(self._portfolio_greeks)

        except Exception as e:
            logger.error(f"Error updating greeks: {e}")

    def _add_to_history(self, greeks: PortfolioGreeks) -> None:
        """添加到历史记录"""
        self._greeks_history.append(greeks)
        if len(self._greeks_history) > self._config.history_size:
            self._greeks_history = self._greeks_history[-self._config.history_size:]

    # ============ Limits and Alerts ============

    def set_limit(self, limit: GreeksLimit) -> None:
        """设置限额"""
        self._limits[limit.greek_type] = limit

    def get_limit(self, greek_type: GreekType) -> Optional[GreeksLimit]:
        """获取限额"""
        return self._limits.get(greek_type)

    def _check_limits(self, portfolio_greeks: PortfolioGreeks) -> None:
        """检查限额"""
        greeks = portfolio_greeks.total_greeks
        greek_values = {
            GreekType.DELTA: greeks.delta,
            GreekType.GAMMA: greeks.gamma,
            GreekType.THETA: greeks.theta,
            GreekType.VEGA: greeks.vega,
            GreekType.RHO: greeks.rho,
        }

        for greek_type, value in greek_values.items():
            limit = self._limits.get(greek_type)
            if not limit:
                continue

            # 检查最大值
            if limit.max_value is not None and value > limit.max_value:
                self._create_alert(
                    greek_type, value, limit.max_value,
                    AlertSeverity.HIGH,
                    f"{greek_type.value} exceeds maximum limit"
                )
            elif limit.max_value is not None and value > limit.max_value * limit.warning_threshold:
                self._create_alert(
                    greek_type, value, limit.max_value,
                    AlertSeverity.MEDIUM,
                    f"{greek_type.value} approaching maximum limit"
                )

            # 检查最小值
            if limit.min_value is not None and value < limit.min_value:
                self._create_alert(
                    greek_type, value, limit.min_value,
                    AlertSeverity.HIGH,
                    f"{greek_type.value} below minimum limit"
                )

    def _create_alert(
        self,
        greek_type: GreekType,
        value: float,
        threshold: float,
        severity: AlertSeverity,
        message: str,
    ) -> None:
        """创建警报"""
        self._alert_counter += 1
        alert = GreeksAlert(
            alert_id=f"GREEK_ALERT_{self._alert_counter}",
            position_id=None,
            greek_type=greek_type,
            current_value=value,
            threshold=threshold,
            severity=severity,
            message=message,
        )
        self._alert_history.append(alert)
        self._notify_alert(alert)

    # ============ Scenario Analysis ============

    def analyze_scenario(
        self,
        position: OptionPosition,
        market_data: MarketData,
        spot_change_pct: float = 0.0,
        vol_change: float = 0.0,
        time_change_days: float = 0.0,
        rate_change: float = 0.0,
    ) -> ScenarioResult:
        """情景分析"""
        now = datetime.now()
        time_to_expiry = max((position.expiry - now).total_seconds() / (365.25 * 24 * 3600), 0)

        # 当前定价
        current_pricing = self.price_option(
            position.option_type,
            market_data.spot_price,
            position.strike,
            time_to_expiry,
            market_data.volatility,
            market_data.risk_free_rate,
            market_data.dividend_yield,
        )

        # 新市场条件
        new_spot = market_data.spot_price * (1 + spot_change_pct)
        new_vol = market_data.volatility + vol_change
        new_time = max(time_to_expiry - time_change_days / 365.0, 0)
        new_rate = market_data.risk_free_rate + rate_change

        # 新定价
        new_pricing = self.price_option(
            position.option_type,
            new_spot,
            position.strike,
            new_time,
            new_vol,
            new_rate,
            market_data.dividend_yield,
        )

        # 计算变化
        price_change = new_pricing.theoretical_price - current_pricing.theoretical_price
        pnl = price_change * position.quantity * position.multiplier

        greeks_change = Greeks(
            delta=new_pricing.greeks.delta - current_pricing.greeks.delta,
            gamma=new_pricing.greeks.gamma - current_pricing.greeks.gamma,
            theta=new_pricing.greeks.theta - current_pricing.greeks.theta,
            vega=new_pricing.greeks.vega - current_pricing.greeks.vega,
            rho=new_pricing.greeks.rho - current_pricing.greeks.rho,
        )

        return ScenarioResult(
            scenario_name=f"Spot:{spot_change_pct:+.1%} Vol:{vol_change:+.2f} Time:{time_change_days}d",
            spot_change=spot_change_pct,
            vol_change=vol_change,
            time_change=time_change_days,
            rate_change=rate_change,
            new_price=new_pricing.theoretical_price,
            price_change=price_change,
            pnl=pnl,
            new_greeks=new_pricing.greeks,
            greeks_change=greeks_change,
        )

    def run_stress_test(
        self,
        position: OptionPosition,
        market_data: MarketData,
    ) -> List[ScenarioResult]:
        """压力测试"""
        scenarios = []

        # 标的价格变化
        for spot_change in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
            result = self.analyze_scenario(
                position, market_data, spot_change_pct=spot_change
            )
            scenarios.append(result)

        # 波动率变化
        for vol_change in [-0.10, -0.05, 0.05, 0.10]:
            result = self.analyze_scenario(
                position, market_data, vol_change=vol_change
            )
            scenarios.append(result)

        # 时间衰减
        for time_change in [1, 7, 30]:
            result = self.analyze_scenario(
                position, market_data, time_change_days=time_change
            )
            scenarios.append(result)

        return scenarios

    # ============ P&L Attribution ============

    def attribute_pnl(
        self,
        position: OptionPosition,
        old_market: MarketData,
        new_market: MarketData,
        old_price: float,
        new_price: float,
        time_passed_days: float,
    ) -> PnLAttribution:
        """P&L归因分析"""
        now = datetime.now()
        time_to_expiry = max((position.expiry - now).total_seconds() / (365.25 * 24 * 3600), 0)

        # 计算旧希腊字母
        old_greeks = BlackScholesPricer.greeks(
            position.option_type,
            old_market.spot_price,
            position.strike,
            time_to_expiry + time_passed_days / 365.0,
            old_market.risk_free_rate,
            old_market.dividend_yield,
            old_market.volatility,
        )

        # 市场变化
        spot_change = new_market.spot_price - old_market.spot_price
        vol_change = new_market.volatility - old_market.volatility
        rate_change = new_market.risk_free_rate - old_market.risk_free_rate

        # 希腊字母归因
        factor = position.quantity * position.multiplier

        delta_pnl = old_greeks.delta * spot_change * factor
        gamma_pnl = 0.5 * old_greeks.gamma * spot_change * spot_change * factor
        theta_pnl = old_greeks.theta * time_passed_days * factor
        vega_pnl = old_greeks.vega * (vol_change * 100) * factor  # vega是1%变化
        rho_pnl = old_greeks.rho * (rate_change * 100) * factor

        total_pnl = (new_price - old_price) * factor
        explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
        unexplained_pnl = total_pnl - explained_pnl

        return PnLAttribution(
            position_id=position.position_id,
            total_pnl=total_pnl,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            rho_pnl=rho_pnl,
            unexplained_pnl=unexplained_pnl,
            spot_change=spot_change,
            vol_change=vol_change,
            time_passed=time_passed_days,
            rate_change=rate_change,
        )

    # ============ Hedging ============

    def calculate_delta_hedge(
        self, portfolio_greeks: PortfolioGreeks
    ) -> Dict[str, float]:
        """计算Delta对冲需求"""
        hedge_orders = {}

        for symbol, greeks in portfolio_greeks.by_underlying.items():
            if abs(greeks.delta) > 0.01:  # 忽略很小的delta
                # 需要卖出的股票数量来对冲
                hedge_shares = -greeks.delta
                hedge_orders[symbol] = hedge_shares

        return hedge_orders

    def calculate_gamma_scalping_levels(
        self,
        position: OptionPosition,
        market_data: MarketData,
        gamma_threshold: float = 0.1,
    ) -> Dict[str, float]:
        """计算Gamma Scalping水平"""
        now = datetime.now()
        time_to_expiry = max((position.expiry - now).total_seconds() / (365.25 * 24 * 3600), 0)

        pricing = self.price_option(
            position.option_type,
            market_data.spot_price,
            position.strike,
            time_to_expiry,
            market_data.volatility,
            market_data.risk_free_rate,
            market_data.dividend_yield,
        )

        gamma = pricing.greeks.gamma
        if abs(gamma) < 0.001:
            return {}

        # 计算需要的价格变动来产生指定delta变化
        delta_change_needed = gamma_threshold
        price_move = delta_change_needed / gamma

        return {
            "current_price": market_data.spot_price,
            "upper_rehedge": market_data.spot_price + price_move,
            "lower_rehedge": market_data.spot_price - price_move,
            "current_gamma": gamma,
            "delta_threshold": gamma_threshold,
        }

    # ============ Callbacks ============

    def on_greeks_update(
        self, callback: Callable[[PortfolioGreeks], None]
    ) -> None:
        """注册希腊字母更新回调"""
        self._greeks_callbacks.append(callback)

    def on_alert(
        self, callback: Callable[[GreeksAlert], None]
    ) -> None:
        """注册警报回调"""
        self._alert_callbacks.append(callback)

    def _notify_greeks(self, greeks: PortfolioGreeks) -> None:
        """通知希腊字母更新"""
        for callback in self._greeks_callbacks:
            try:
                callback(greeks)
            except Exception as e:
                logger.error(f"Error in greeks callback: {e}")

    def _notify_alert(self, alert: GreeksAlert) -> None:
        """通知警报"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    # ============ Public API ============

    def get_position_greeks(self, position_id: str) -> Optional[PositionGreeks]:
        """获取持仓希腊字母"""
        return self._position_greeks.get(position_id)

    def get_portfolio_greeks(self) -> Optional[PortfolioGreeks]:
        """获取组合希腊字母"""
        return self._portfolio_greeks

    def get_greeks_history(self, limit: int = 100) -> List[PortfolioGreeks]:
        """获取希腊字母历史"""
        return self._greeks_history[-limit:]

    def get_alerts(self, limit: int = 100) -> List[GreeksAlert]:
        """获取警报列表"""
        return self._alert_history[-limit:]

    def get_state(self) -> CalculatorState:
        """获取计算器状态"""
        return self._state

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "positions_tracked": len(self._position_greeks),
            "total_alerts": len(self._alert_history),
            "history_size": len(self._greeks_history),
            "limits_configured": len(self._limits),
            "portfolio_greeks": self._portfolio_greeks.to_dict() if self._portfolio_greeks else None,
        }


# ============ Factory Functions ============


def create_greeks_calculator(
    pricing_model: str = "black_scholes",
    calculate_higher_order: bool = True,
    market_provider: Optional[MarketDataProvider] = None,
    position_provider: Optional[PositionProvider] = None,
) -> GreeksCalculator:
    """创建希腊字母计算器"""
    config = CalculatorConfig(
        pricing_model=PricingModel(pricing_model),
        calculate_higher_order=calculate_higher_order,
    )
    return GreeksCalculator(
        config=config,
        market_provider=market_provider,
        position_provider=position_provider,
    )


def create_option_position(
    position_id: str,
    symbol: str,
    option_symbol: str,
    option_type: str,
    strike: float,
    expiry: datetime,
    quantity: int,
    multiplier: float = 100.0,
) -> OptionPosition:
    """创建期权持仓"""
    return OptionPosition(
        position_id=position_id,
        symbol=symbol,
        option_symbol=option_symbol,
        option_type=OptionType(option_type),
        option_style=OptionStyle.EUROPEAN,
        strike=strike,
        expiry=expiry,
        quantity=quantity,
        multiplier=multiplier,
    )


def create_market_data(
    symbol: str,
    spot_price: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> MarketData:
    """创建市场数据"""
    return MarketData(
        symbol=symbol,
        spot_price=spot_price,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def quick_greeks(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """快速计算希腊字母"""
    greeks = BlackScholesPricer.greeks(
        OptionType(option_type),
        spot, strike, time_to_expiry,
        risk_free_rate, dividend_yield, volatility,
        calculate_higher_order=False,
    )
    return greeks.to_dict()


# ============ Exports ============


__all__ = [
    # Enums
    "OptionType",
    "OptionStyle",
    "PricingModel",
    "GreekType",
    "AlertSeverity",
    "CalculatorState",
    # Data Classes
    "OptionPosition",
    "MarketData",
    "Greeks",
    "OptionPricing",
    "PositionGreeks",
    "PortfolioGreeks",
    "ScenarioResult",
    "PnLAttribution",
    "GreeksAlert",
    "GreeksLimit",
    "CalculatorConfig",
    # Interfaces
    "MarketDataProvider",
    "PositionProvider",
    # Simulated Providers
    "SimulatedMarketDataProvider",
    "SimulatedPositionProvider",
    # Pricing Models
    "BlackScholesPricer",
    # Main Class
    "GreeksCalculator",
    # Factory Functions
    "create_greeks_calculator",
    "create_option_position",
    "create_market_data",
    "quick_greeks",
    # Utilities
    "norm_cdf",
    "norm_pdf",
]
