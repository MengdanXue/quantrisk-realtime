"""
Option Pricing Engine - 期权定价引擎

提供全面的期权定价和风险分析功能，支持多种定价模型和期权类型。

核心功能：
1. Black-Scholes 欧式期权定价模型
2. Binomial Tree 二叉树定价（美式期权）
3. Monte Carlo 蒙特卡洛模拟定价
4. 隐含波动率计算（Newton-Raphson迭代）
5. 期权Greeks计算（Delta, Gamma, Theta, Vega, Rho）
6. LSM 最小二乘蒙特卡洛（美式期权）
7. 障碍期权定价
8. 期权组合定价
9. 波动率曲面拟合
10. 定价缓存与批量计算优化

Copyright (c) 2025 TraderX. All rights reserved.
"""

import asyncio
import logging
import math
import statistics
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============ Constants ============

SQRT_2PI = math.sqrt(2 * math.pi)
DAYS_PER_YEAR = 365.25


# ============ Enums ============

class OptionType(str, Enum):
    """期权类型"""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """期权风格"""
    EUROPEAN = "european"  # 欧式期权
    AMERICAN = "american"  # 美式期权


class BarrierType(str, Enum):
    """障碍期权类型"""
    UP_AND_IN = "up_and_in"      # 上升敲入
    UP_AND_OUT = "up_and_out"    # 上升敲出
    DOWN_AND_IN = "down_and_in"  # 下降敲入
    DOWN_AND_OUT = "down_and_out"  # 下降敲出


class PricingModel(str, Enum):
    """定价模型"""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL_TREE = "binomial_tree"
    MONTE_CARLO = "monte_carlo"
    LSM = "lsm"  # Longstaff-Schwartz Method
    BARRIER = "barrier"
    FINITE_DIFFERENCE = "finite_difference"


class VolatilitySurfaceModel(str, Enum):
    """波动率曲面模型"""
    SVI = "svi"  # Stochastic Volatility Inspired
    SABR = "sabr"  # Stochastic Alpha Beta Rho
    POLYNOMIAL = "polynomial"
    LINEAR_INTERPOLATION = "linear_interpolation"


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class CacheStrategy(str, Enum):
    """缓存策略"""
    NONE = "none"
    LRU = "lru"  # Least Recently Used
    TIME_BASED = "time_based"  # 基于时间过期


# ============ Math Utilities ============

def norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数"""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_inv(p: float) -> float:
    """
    标准正态分布逆累积分布函数（近似）

    使用Acklam算法

    Args:
        p: 概率值 (0, 1)

    Returns:
        对应的z值
    """
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be in (0, 1)")

    # Acklam算法系数
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]

    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]

    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]

    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    # 定义分界点
    p_low = 0.02425
    p_high = 1 - p_low

    # 低区域
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # 中心区域
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

    # 高区域 - 使用对称性: norm_inv(1-p) = -norm_inv(p)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    return x


# ============ Data Classes ============

@dataclass
class OptionContract:
    """
    期权合约

    定义一个完整的期权合约信息
    """
    contract_id: str
    underlying_symbol: str           # 标的资产代码
    option_symbol: str               # 期权代码
    option_type: OptionType          # 期权类型
    option_style: OptionStyle        # 期权风格
    strike: float                    # 行权价
    expiry: datetime                 # 到期日
    multiplier: float = 100.0        # 合约乘数

    # 障碍期权参数（可选）
    barrier_type: Optional[BarrierType] = None
    barrier_level: Optional[float] = None
    rebate: float = 0.0              # 敲出补偿

    def time_to_expiry(self, current_time: Optional[datetime] = None) -> float:
        """
        计算到期时间（年）

        Args:
            current_time: 当前时间，默认为系统时间

        Returns:
            到期时间（年）
        """
        current_time = current_time or datetime.now()
        delta = (self.expiry - current_time).total_seconds()
        return max(delta / (DAYS_PER_YEAR * 24 * 3600), 0.0)

    def is_call(self) -> bool:
        """是否为看涨期权"""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """是否为看跌期权"""
        return self.option_type == OptionType.PUT

    def is_european(self) -> bool:
        """是否为欧式期权"""
        return self.option_style == OptionStyle.EUROPEAN

    def is_american(self) -> bool:
        """是否为美式期权"""
        return self.option_style == OptionStyle.AMERICAN

    def is_barrier(self) -> bool:
        """是否为障碍期权"""
        return self.barrier_type is not None

    def intrinsic_value(self, spot: float) -> float:
        """
        计算内在价值

        Args:
            spot: 标的价格

        Returns:
            内在价值
        """
        if self.is_call():
            return max(spot - self.strike, 0.0)
        else:
            return max(self.strike - spot, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "underlying_symbol": self.underlying_symbol,
            "option_symbol": self.option_symbol,
            "option_type": self.option_type.value,
            "option_style": self.option_style.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "multiplier": self.multiplier,
            "barrier_type": self.barrier_type.value if self.barrier_type else None,
            "barrier_level": self.barrier_level,
            "rebate": self.rebate,
        }


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    spot_price: float                # 现货价格
    volatility: float                # 波动率
    risk_free_rate: float            # 无风险利率
    dividend_yield: float = 0.0      # 股息率
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
class GreeksResult:
    """
    Greeks结果

    期权风险指标
    """
    delta: float = 0.0       # 标的价格敏感度
    gamma: float = 0.0       # Delta的敏感度
    theta: float = 0.0       # 时间衰减（每日）
    vega: float = 0.0        # 波动率敏感度（1%变化）
    rho: float = 0.0         # 利率敏感度（1%变化）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }

    def scale(self, factor: float) -> "GreeksResult":
        """按因子缩放"""
        return GreeksResult(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
        )

    def add(self, other: "GreeksResult") -> "GreeksResult":
        """相加"""
        return GreeksResult(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
        )


@dataclass
class PricingResult:
    """
    定价结果

    包含期权理论价格、Greeks和其他信息
    """
    contract_id: str
    theoretical_price: float         # 理论价格
    intrinsic_value: float           # 内在价值
    time_value: float                # 时间价值
    greeks: GreeksResult             # Greeks
    model: PricingModel              # 使用的模型
    timestamp: datetime = field(default_factory=datetime.now)

    # 模型特定信息
    model_params: Dict[str, Any] = field(default_factory=dict)
    convergence_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "theoretical_price": self.theoretical_price,
            "intrinsic_value": self.intrinsic_value,
            "time_value": self.time_value,
            "greeks": self.greeks.to_dict(),
            "model": self.model.value,
            "timestamp": self.timestamp.isoformat(),
            "model_params": self.model_params,
            "convergence_info": self.convergence_info,
        }


@dataclass
class VolatilityPoint:
    """波动率点"""
    strike: float
    expiry: datetime
    implied_volatility: float

    def time_to_expiry(self, current_time: Optional[datetime] = None) -> float:
        """计算到期时间（年）"""
        current_time = current_time or datetime.now()
        delta = (self.expiry - current_time).total_seconds()
        return max(delta / (DAYS_PER_YEAR * 24 * 3600), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "implied_volatility": self.implied_volatility,
        }


@dataclass
class VolatilitySurface:
    """
    波动率曲面

    存储和插值期权隐含波动率
    """
    underlying_symbol: str
    points: List[VolatilityPoint]
    surface_model: VolatilitySurfaceModel
    timestamp: datetime = field(default_factory=datetime.now)

    # 模型参数
    model_params: Dict[str, Any] = field(default_factory=dict)

    def get_volatility(
        self,
        strike: float,
        expiry: datetime,
        spot: float,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        获取特定行权价和到期日的隐含波动率

        Args:
            strike: 行权价
            expiry: 到期日
            spot: 标的价格
            current_time: 当前时间

        Returns:
            隐含波动率
        """
        current_time = current_time or datetime.now()

        if not self.points:
            return 0.2  # 默认波动率

        # 计算目标到期时间
        target_ttm = (expiry - current_time).total_seconds() / (DAYS_PER_YEAR * 24 * 3600)
        target_ttm = max(target_ttm, 0.0)

        # 计算moneyness (strike/spot)
        moneyness = strike / spot if spot > 0 else 1.0

        # 线性插值（简化实现）
        if self.surface_model == VolatilitySurfaceModel.LINEAR_INTERPOLATION:
            return self._linear_interpolation(moneyness, target_ttm)

        # 其他模型暂时使用最近点
        return self._nearest_neighbor(moneyness, target_ttm)

    def _linear_interpolation(self, moneyness: float, ttm: float) -> float:
        """线性插值"""
        if not self.points:
            return 0.2

        # 找到最接近的点
        closest_point = min(
            self.points,
            key=lambda p: abs(p.strike / moneyness - 1.0) + abs(p.time_to_expiry() - ttm)
        )

        return closest_point.implied_volatility

    def _nearest_neighbor(self, moneyness: float, ttm: float) -> float:
        """最近邻"""
        if not self.points:
            return 0.2

        closest_point = min(
            self.points,
            key=lambda p: abs(p.strike / moneyness - 1.0) + abs(p.time_to_expiry() - ttm)
        )

        return closest_point.implied_volatility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "underlying_symbol": self.underlying_symbol,
            "points": [p.to_dict() for p in self.points],
            "surface_model": self.surface_model.value,
            "timestamp": self.timestamp.isoformat(),
            "model_params": self.model_params,
        }


@dataclass
class PortfolioPricing:
    """组合定价结果"""
    portfolio_id: str
    contracts: List[OptionContract]
    pricing_results: List[PricingResult]
    total_value: float
    total_greeks: GreeksResult
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "contracts": [c.to_dict() for c in self.contracts],
            "pricing_results": [p.to_dict() for p in self.pricing_results],
            "total_value": self.total_value,
            "total_greeks": self.total_greeks.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PricingConfig:
    """定价配置"""
    # 通用配置
    default_model: PricingModel = PricingModel.BLACK_SCHOLES
    enable_greeks: bool = True
    enable_cache: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.TIME_BASED
    cache_ttl_seconds: int = 60

    # 二叉树配置
    binomial_steps: int = 100        # 二叉树步数

    # 蒙特卡洛配置
    mc_simulations: int = 10000      # 模拟路径数
    mc_time_steps: int = 50          # 时间步数
    mc_seed: Optional[int] = None    # 随机种子
    mc_antithetic: bool = True       # 对偶变量法

    # LSM配置
    lsm_simulations: int = 10000     # LSM模拟路径数
    lsm_basis_degree: int = 2        # 基函数多项式阶数

    # 隐含波动率计算
    iv_max_iterations: int = 100     # 最大迭代次数
    iv_tolerance: float = 1e-6       # 收敛容差
    iv_initial_guess: float = 0.2    # 初始猜测

    # 数值稳定性
    min_time_to_expiry: float = 1e-6  # 最小到期时间（年）
    min_volatility: float = 0.01      # 最小波动率
    max_volatility: float = 5.0       # 最大波动率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_model": self.default_model.value,
            "enable_greeks": self.enable_greeks,
            "enable_cache": self.enable_cache,
            "cache_strategy": self.cache_strategy.value,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "binomial_steps": self.binomial_steps,
            "mc_simulations": self.mc_simulations,
            "mc_time_steps": self.mc_time_steps,
            "mc_antithetic": self.mc_antithetic,
            "lsm_simulations": self.lsm_simulations,
            "lsm_basis_degree": self.lsm_basis_degree,
            "iv_max_iterations": self.iv_max_iterations,
            "iv_tolerance": self.iv_tolerance,
        }


# ============ Pricing Models ============

class BlackScholesModel:
    """
    Black-Scholes 定价模型

    用于欧式期权定价
    """

    @staticmethod
    def calculate_d1_d2(
        S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> Tuple[float, float]:
        """
        计算 d1 和 d2

        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率

        Returns:
            (d1, d2)
        """
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return d1, d2

    @staticmethod
    def price(
        option_type: OptionType,
        S: float, K: float, T: float,
        r: float, q: float, sigma: float
    ) -> float:
        """
        计算期权价格

        Args:
            option_type: 期权类型
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率

        Returns:
            期权价格
        """
        if T <= 0:
            # 到期时的内在价值
            if option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, q, sigma)

        if option_type == OptionType.CALL:
            price = S * math.exp(-q * T) * norm_cdf(d1) - \
                    K * math.exp(-r * T) * norm_cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm_cdf(-d2) - \
                    S * math.exp(-q * T) * norm_cdf(-d1)

        return max(price, 0.0)

    @staticmethod
    def greeks(
        option_type: OptionType,
        S: float, K: float, T: float,
        r: float, q: float, sigma: float
    ) -> GreeksResult:
        """
        计算Greeks

        Args:
            option_type: 期权类型
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率

        Returns:
            Greeks结果
        """
        if T <= 0 or sigma <= 0:
            # 到期或无效波动率
            delta = 1.0 if (option_type == OptionType.CALL and S > K) else 0.0
            if option_type == OptionType.PUT and S < K:
                delta = -1.0
            return GreeksResult(delta=delta)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, q, sigma)
        sqrt_T = math.sqrt(T)

        # 基本Greeks
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
        theta = (term1 + term2 + term3) / DAYS_PER_YEAR

        # Vega (1%波动率变化)
        vega = S * math.exp(-q * T) * sqrt_T * nd1 / 100

        # Rho (1%利率变化)
        if option_type == OptionType.CALL:
            rho = K * T * math.exp(-r * T) * Nd2 / 100
        else:
            rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )


class BinomialTreeModel:
    """
    二叉树定价模型

    用于美式期权定价
    """

    @staticmethod
    def price(
        option_type: OptionType,
        S: float, K: float, T: float,
        r: float, q: float, sigma: float,
        steps: int = 100,
        american: bool = True
    ) -> float:
        """
        二叉树定价

        Args:
            option_type: 期权类型
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率
            steps: 时间步数
            american: 是否为美式期权

        Returns:
            期权价格
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))  # 上涨因子
        d = 1 / u  # 下跌因子
        p = (math.exp((r - q) * dt) - d) / (u - d)  # 风险中性概率
        discount = math.exp(-r * dt)

        # 构建价格树（仅需最后一层）
        prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            prices[i] = S * (u ** (steps - i)) * (d ** i)

        # 计算到期日期权价值
        if option_type == OptionType.CALL:
            values = np.maximum(prices - K, 0.0)
        else:
            values = np.maximum(K - prices, 0.0)

        # 向后递推
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                # 持有价值（期望现值）
                hold_value = discount * (p * values[i] + (1 - p) * values[i + 1])

                if american:
                    # 美式期权：取持有价值和立即行权价值的最大值
                    spot = S * (u ** (step - i)) * (d ** i)
                    if option_type == OptionType.CALL:
                        exercise_value = max(spot - K, 0.0)
                    else:
                        exercise_value = max(K - spot, 0.0)
                    values[i] = max(hold_value, exercise_value)
                else:
                    # 欧式期权：只能持有
                    values[i] = hold_value

        return values[0]


class MonteCarloModel:
    """
    蒙特卡洛模拟定价模型

    用于路径依赖期权和复杂期权定价
    """

    @staticmethod
    def price(
        option_type: OptionType,
        S: float, K: float, T: float,
        r: float, q: float, sigma: float,
        simulations: int = 10000,
        time_steps: int = 50,
        seed: Optional[int] = None,
        antithetic: bool = True
    ) -> float:
        """
        蒙特卡洛定价（欧式期权）

        Args:
            option_type: 期权类型
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率
            simulations: 模拟路径数
            time_steps: 时间步数
            seed: 随机种子
            antithetic: 是否使用对偶变量法

        Returns:
            期权价格
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        if seed is not None:
            np.random.seed(seed)

        dt = T / time_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        # 生成随机数
        if antithetic:
            # 对偶变量法：减少方差
            half_sims = simulations // 2
            Z = np.random.standard_normal((half_sims, time_steps))
            Z_full = np.vstack([Z, -Z])[:simulations]
        else:
            Z_full = np.random.standard_normal((simulations, time_steps))

        # 模拟价格路径
        S_T = S * np.exp(np.sum(drift + diffusion * Z_full, axis=1))

        # 计算payoff
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        # 折现
        price = math.exp(-r * T) * np.mean(payoffs)

        return price


class LSMModel:
    """
    Longstaff-Schwartz Method (LSM)

    用于美式期权定价（最小二乘蒙特卡洛）
    """

    @staticmethod
    def price(
        option_type: OptionType,
        S: float, K: float, T: float,
        r: float, q: float, sigma: float,
        simulations: int = 10000,
        time_steps: int = 50,
        basis_degree: int = 2,
        seed: Optional[int] = None
    ) -> float:
        """
        LSM美式期权定价

        Args:
            option_type: 期权类型
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率
            simulations: 模拟路径数
            time_steps: 时间步数
            basis_degree: 基函数多项式阶数
            seed: 随机种子

        Returns:
            期权价格
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        if seed is not None:
            np.random.seed(seed)

        dt = T / time_steps
        discount = math.exp(-r * dt)
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        # 生成价格路径
        Z = np.random.standard_normal((simulations, time_steps))
        paths = np.zeros((simulations, time_steps + 1))
        paths[:, 0] = S

        for t in range(time_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

        # 计算到期日payoff
        if option_type == OptionType.CALL:
            cashflows = np.maximum(paths[:, -1] - K, 0.0)
        else:
            cashflows = np.maximum(K - paths[:, -1], 0.0)

        # 向后递推（LSM核心算法）
        for t in range(time_steps - 1, 0, -1):
            # 当前时刻标的价格
            S_t = paths[:, t]

            # 立即行权价值
            if option_type == OptionType.CALL:
                exercise_value = np.maximum(S_t - K, 0.0)
            else:
                exercise_value = np.maximum(K - S_t, 0.0)

            # 只考虑实值期权
            in_the_money = exercise_value > 0

            if np.sum(in_the_money) > 0:
                # 构建基函数矩阵（拉盖尔多项式）
                X = S_t[in_the_money]
                Y = cashflows[in_the_money] * discount

                # 多项式回归
                basis = np.vstack([X ** i for i in range(basis_degree + 1)]).T

                try:
                    # 最小二乘拟合
                    coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
                    continuation_value = basis @ coeffs

                    # 判断是否提前行权
                    exercise_now = exercise_value[in_the_money] > continuation_value

                    # 更新现金流
                    cashflows[in_the_money] = np.where(
                        exercise_now,
                        exercise_value[in_the_money],
                        cashflows[in_the_money] * discount
                    )
                except np.linalg.LinAlgError:
                    # 回归失败，保持原现金流
                    cashflows[in_the_money] *= discount
            else:
                # 全部虚值，折现
                cashflows *= discount

        # 初始时刻的期权价值
        price = discount * np.mean(cashflows)

        return price


class BarrierModel:
    """
    障碍期权定价模型

    基于解析解或蒙特卡洛模拟
    """

    @staticmethod
    def price(
        option_type: OptionType,
        barrier_type: BarrierType,
        S: float, K: float, H: float, T: float,
        r: float, q: float, sigma: float,
        rebate: float = 0.0
    ) -> float:
        """
        障碍期权定价

        Args:
            option_type: 期权类型
            barrier_type: 障碍类型
            S: 标的价格
            K: 行权价
            H: 障碍价格
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            sigma: 波动率
            rebate: 敲出补偿

        Returns:
            期权价格
        """
        # 简化实现：使用Black-Scholes价格调整
        # 完整实现需要更复杂的解析公式

        vanilla_price = BlackScholesModel.price(option_type, S, K, T, r, q, sigma)

        # 根据障碍类型调整
        if barrier_type == BarrierType.UP_AND_OUT:
            if S >= H:
                return rebate
            # 简化：价格衰减
            adjustment = math.exp(-2 * (H - S) / (sigma * S * math.sqrt(T)))
            return vanilla_price * (1 - adjustment) + rebate * adjustment

        elif barrier_type == BarrierType.DOWN_AND_OUT:
            if S <= H:
                return rebate
            adjustment = math.exp(-2 * (S - H) / (sigma * S * math.sqrt(T)))
            return vanilla_price * (1 - adjustment) + rebate * adjustment

        elif barrier_type == BarrierType.UP_AND_IN:
            if S >= H:
                return vanilla_price
            adjustment = math.exp(-2 * (H - S) / (sigma * S * math.sqrt(T)))
            return vanilla_price * adjustment

        elif barrier_type == BarrierType.DOWN_AND_IN:
            if S <= H:
                return vanilla_price
            adjustment = math.exp(-2 * (S - H) / (sigma * S * math.sqrt(T)))
            return vanilla_price * adjustment

        return vanilla_price


# ============ Implied Volatility Calculator ============

class ImpliedVolatilityCalculator:
    """
    隐含波动率计算器

    使用Newton-Raphson迭代法求解隐含波动率
    """

    @staticmethod
    def calculate(
        option_type: OptionType,
        market_price: float,
        S: float, K: float, T: float,
        r: float, q: float,
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Tuple[float, bool, int]:
        """
        计算隐含波动率

        Args:
            option_type: 期权类型
            market_price: 市场价格
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            q: 股息率
            initial_guess: 初始猜测
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            (隐含波动率, 是否收敛, 迭代次数)
        """
        if T <= 0 or market_price <= 0:
            return 0.0, False, 0

        # 检查内在价值
        intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        if market_price < intrinsic:
            return 0.0, False, 0

        sigma = initial_guess

        for iteration in range(max_iterations):
            # 计算理论价格
            price = BlackScholesModel.price(option_type, S, K, T, r, q, sigma)

            # 计算Vega（对波动率的导数）
            d1, _ = BlackScholesModel.calculate_d1_d2(S, K, T, r, q, sigma)
            vega = S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)

            # 检查数值稳定性
            if abs(vega) < 1e-10:
                break

            # Newton-Raphson迭代
            diff = market_price - price
            sigma_new = sigma + diff / vega

            # 限制波动率范围
            sigma_new = max(0.001, min(sigma_new, 5.0))

            # 检查收敛
            if abs(sigma_new - sigma) < tolerance:
                return sigma_new, True, iteration + 1

            sigma = sigma_new

        # 未收敛
        return sigma, False, max_iterations


# ============ Pricing Cache ============

class PricingCache:
    """定价结果缓存"""

    def __init__(self, strategy: CacheStrategy, ttl_seconds: int = 60):
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[PricingResult, datetime]] = {}
        self._access_order: deque = deque(maxlen=1000)  # LRU

    def _make_key(
        self,
        contract_id: str,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        model: PricingModel
    ) -> str:
        """生成缓存键"""
        return f"{contract_id}:{spot:.2f}:{volatility:.4f}:{risk_free_rate:.4f}:{model.value}"

    def get(
        self,
        contract_id: str,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        model: PricingModel
    ) -> Optional[PricingResult]:
        """获取缓存的定价结果"""
        if self.strategy == CacheStrategy.NONE:
            return None

        key = self._make_key(contract_id, spot, volatility, risk_free_rate, model)

        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]

        # 检查过期
        if self.strategy == CacheStrategy.TIME_BASED:
            age = (datetime.now() - timestamp).total_seconds()
            if age > self.ttl_seconds:
                del self._cache[key]
                return None

        # LRU更新
        if self.strategy == CacheStrategy.LRU:
            self._access_order.append(key)

        return result

    def put(
        self,
        contract_id: str,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        model: PricingModel,
        result: PricingResult
    ) -> None:
        """缓存定价结果"""
        if self.strategy == CacheStrategy.NONE:
            return

        key = self._make_key(contract_id, spot, volatility, risk_free_rate, model)
        self._cache[key] = (result, datetime.now())

        if self.strategy == CacheStrategy.LRU:
            self._access_order.append(key)

            # LRU淘汰
            if len(self._cache) > 1000:
                oldest_key = self._access_order[0]
                if oldest_key in self._cache:
                    del self._cache[oldest_key]

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()


# ============ Main Engine ============

class OptionPricingEngine:
    """
    期权定价引擎

    提供完整的期权定价、Greeks计算和风险分析功能
    """

    def __init__(self, config: Optional[PricingConfig] = None):
        self.config = config or PricingConfig()

        # 状态
        self._state = EngineState.IDLE

        # 缓存
        self._cache = PricingCache(
            self.config.cache_strategy,
            self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None

        # 波动率曲面
        self._volatility_surfaces: Dict[str, VolatilitySurface] = {}

        # 统计信息
        self._pricing_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("OptionPricingEngine initialized")

    # ============ Core Pricing Methods ============

    def _price_only(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model: PricingModel
    ) -> float:
        """
        纯定价方法（不计算Greeks），用于有限差分计算

        Args:
            contract: 期权合约
            market_data: 市场数据
            model: 定价模型

        Returns:
            期权价格
        """
        T = contract.time_to_expiry()
        if T < self.config.min_time_to_expiry:
            return contract.intrinsic_value(market_data.spot_price)

        T = max(T, self.config.min_time_to_expiry)
        sigma = max(
            self.config.min_volatility,
            min(market_data.volatility, self.config.max_volatility)
        )

        if model == PricingModel.BLACK_SCHOLES:
            return BlackScholesModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            )
        elif model == PricingModel.MONTE_CARLO:
            return MonteCarloModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma,
                simulations=self.config.mc_simulations,
                time_steps=self.config.mc_time_steps,
                seed=self.config.mc_seed,
                antithetic=self.config.mc_antithetic
            )
        else:
            # 默认Black-Scholes
            return BlackScholesModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            )

    def price_european_option(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model: Optional[PricingModel] = None
    ) -> PricingResult:
        """
        欧式期权定价

        Args:
            contract: 期权合约
            market_data: 市场数据
            model: 定价模型（可选）

        Returns:
            定价结果
        """
        model = model or self.config.default_model

        # 检查缓存
        if self._cache:
            cached = self._cache.get(
                contract.contract_id,
                market_data.spot_price,
                market_data.volatility,
                market_data.risk_free_rate,
                model
            )
            if cached:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        # 计算到期时间
        T = contract.time_to_expiry()

        # 如果已到期或接近到期，直接返回内在价值
        if T < self.config.min_time_to_expiry:
            intrinsic = contract.intrinsic_value(market_data.spot_price)
            result = PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=intrinsic,
                intrinsic_value=intrinsic,
                time_value=0.0,
                greeks=GreeksResult(),
                model=model,
            )
            self._pricing_count += 1
            return result

        T = max(T, self.config.min_time_to_expiry)

        # 限制波动率
        sigma = max(
            self.config.min_volatility,
            min(market_data.volatility, self.config.max_volatility)
        )

        # 根据模型定价
        if model == PricingModel.BLACK_SCHOLES:
            price = BlackScholesModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            )

            greeks = BlackScholesModel.greeks(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            ) if self.config.enable_greeks else GreeksResult()

        elif model == PricingModel.MONTE_CARLO:
            price = MonteCarloModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma,
                simulations=self.config.mc_simulations,
                time_steps=self.config.mc_time_steps,
                seed=self.config.mc_seed,
                antithetic=self.config.mc_antithetic
            )

            # MC的Greeks使用Black-Scholes近似（避免递归）
            greeks = BlackScholesModel.greeks(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            ) if self.config.enable_greeks else GreeksResult()

        else:
            # 默认Black-Scholes
            price = BlackScholesModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            )
            greeks = GreeksResult()

        # 计算内在价值和时间价值
        intrinsic = contract.intrinsic_value(market_data.spot_price)
        time_value = max(price - intrinsic, 0.0)

        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            greeks=greeks,
            model=model,
        )

        # 缓存结果
        if self._cache:
            self._cache.put(
                contract.contract_id,
                market_data.spot_price,
                market_data.volatility,
                market_data.risk_free_rate,
                model,
                result
            )

        self._pricing_count += 1

        return result

    def price_american_option(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model: Optional[PricingModel] = None
    ) -> PricingResult:
        """
        美式期权定价

        Args:
            contract: 期权合约
            market_data: 市场数据
            model: 定价模型（可选，默认使用LSM）

        Returns:
            定价结果
        """
        model = model or PricingModel.LSM

        T = contract.time_to_expiry()
        T = max(T, self.config.min_time_to_expiry)

        sigma = max(
            self.config.min_volatility,
            min(market_data.volatility, self.config.max_volatility)
        )

        # 美式期权定价
        if model == PricingModel.BINOMIAL_TREE:
            price = BinomialTreeModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma,
                steps=self.config.binomial_steps,
                american=True
            )

        elif model == PricingModel.LSM:
            price = LSMModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma,
                simulations=self.config.lsm_simulations,
                time_steps=self.config.mc_time_steps,
                basis_degree=self.config.lsm_basis_degree,
                seed=self.config.mc_seed
            )

        else:
            # 默认LSM
            price = LSMModel.price(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma,
                simulations=self.config.lsm_simulations
            )

        # Greeks（美式期权使用有限差分）
        greeks = self._calculate_greeks_finite_difference(
            contract, market_data, model
        ) if self.config.enable_greeks else GreeksResult()

        intrinsic = contract.intrinsic_value(market_data.spot_price)
        time_value = max(price - intrinsic, 0.0)

        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            greeks=greeks,
            model=model,
        )

        self._pricing_count += 1

        return result

    def price_barrier_option(
        self,
        contract: OptionContract,
        market_data: MarketData
    ) -> PricingResult:
        """
        障碍期权定价

        Args:
            contract: 期权合约（必须包含障碍信息）
            market_data: 市场数据

        Returns:
            定价结果
        """
        if not contract.is_barrier() or contract.barrier_level is None:
            raise ValueError("Contract must have barrier information")

        T = contract.time_to_expiry()
        T = max(T, self.config.min_time_to_expiry)

        sigma = max(
            self.config.min_volatility,
            min(market_data.volatility, self.config.max_volatility)
        )

        price = BarrierModel.price(
            contract.option_type,
            contract.barrier_type,
            market_data.spot_price,
            contract.strike,
            contract.barrier_level,
            T,
            market_data.risk_free_rate,
            market_data.dividend_yield,
            sigma,
            rebate=contract.rebate
        )

        greeks = self._calculate_greeks_finite_difference(
            contract, market_data, PricingModel.BARRIER
        ) if self.config.enable_greeks else GreeksResult()

        intrinsic = contract.intrinsic_value(market_data.spot_price)
        time_value = max(price - intrinsic, 0.0)

        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            greeks=greeks,
            model=PricingModel.BARRIER,
        )

        self._pricing_count += 1

        return result

    def calculate_implied_volatility(
        self,
        contract: OptionContract,
        market_price: float,
        market_data: MarketData
    ) -> Tuple[float, bool]:
        """
        计算隐含波动率

        Args:
            contract: 期权合约
            market_price: 市场价格
            market_data: 市场数据

        Returns:
            (隐含波动率, 是否收敛)
        """
        T = contract.time_to_expiry()

        if T <= 0:
            return 0.0, False

        iv, converged, _ = ImpliedVolatilityCalculator.calculate(
            contract.option_type,
            market_price,
            market_data.spot_price,
            contract.strike,
            T,
            market_data.risk_free_rate,
            market_data.dividend_yield,
            initial_guess=self.config.iv_initial_guess,
            max_iterations=self.config.iv_max_iterations,
            tolerance=self.config.iv_tolerance
        )

        return iv, converged

    def calculate_greeks(
        self,
        contract: OptionContract,
        market_data: MarketData
    ) -> GreeksResult:
        """
        计算期权Greeks

        Args:
            contract: 期权合约
            market_data: 市场数据

        Returns:
            Greeks结果
        """
        T = contract.time_to_expiry()
        T = max(T, self.config.min_time_to_expiry)

        sigma = max(
            self.config.min_volatility,
            min(market_data.volatility, self.config.max_volatility)
        )

        if contract.is_european() and not contract.is_barrier():
            return BlackScholesModel.greeks(
                contract.option_type,
                market_data.spot_price,
                contract.strike,
                T,
                market_data.risk_free_rate,
                market_data.dividend_yield,
                sigma
            )
        else:
            # 美式期权或障碍期权使用有限差分
            return self._calculate_greeks_finite_difference(
                contract, market_data, self.config.default_model
            )

    def _calculate_greeks_finite_difference(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model: PricingModel
    ) -> GreeksResult:
        """
        使用有限差分法计算Greeks

        Args:
            contract: 期权合约
            market_data: 市场数据
            model: 定价模型

        Returns:
            Greeks结果
        """
        # 基准价格 - 使用纯定价方法避免递归
        base_price = self._price_only(contract, market_data, model)

        # Delta: dV/dS
        dS = market_data.spot_price * 0.01
        market_up = MarketData(
            market_data.symbol,
            market_data.spot_price + dS,
            market_data.volatility,
            market_data.risk_free_rate,
            market_data.dividend_yield
        )
        price_up = self._price_only(contract, market_up, model)
        delta = (price_up - base_price) / dS

        # Gamma: d²V/dS²
        market_down = MarketData(
            market_data.symbol,
            market_data.spot_price - dS,
            market_data.volatility,
            market_data.risk_free_rate,
            market_data.dividend_yield
        )
        price_down = self._price_only(contract, market_down, model)
        gamma = (price_up - 2 * base_price + price_down) / (dS * dS)

        # Vega: dV/dσ (1% change)
        dsigma = 0.01
        market_vol_up = MarketData(
            market_data.symbol,
            market_data.spot_price,
            market_data.volatility + dsigma,
            market_data.risk_free_rate,
            market_data.dividend_yield
        )
        price_vol_up = self._price_only(contract, market_vol_up, model)
        vega = price_vol_up - base_price

        # Theta: dV/dt (1 day)
        dt = 1.0 / DAYS_PER_YEAR
        contract_theta = OptionContract(
            contract.contract_id,
            contract.underlying_symbol,
            contract.option_symbol,
            contract.option_type,
            contract.option_style,
            contract.strike,
            contract.expiry - timedelta(days=1),
            contract.multiplier,
            contract.barrier_type,
            contract.barrier_level,
            contract.rebate
        )
        price_time_forward = self._price_only(contract_theta, market_data, model)
        theta = (price_time_forward - base_price) / dt / DAYS_PER_YEAR

        # Rho: dV/dr (1% change)
        dr = 0.01
        market_rate_up = MarketData(
            market_data.symbol,
            market_data.spot_price,
            market_data.volatility,
            market_data.risk_free_rate + dr,
            market_data.dividend_yield
        )
        price_rate_up = self._price_only(contract, market_rate_up, model)
        rho = price_rate_up - base_price

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

    # ============ Portfolio Pricing ============

    def price_portfolio(
        self,
        contracts: List[OptionContract],
        market_data_dict: Dict[str, MarketData],
        quantities: Optional[Dict[str, int]] = None
    ) -> PortfolioPricing:
        """
        期权组合定价

        Args:
            contracts: 期权合约列表
            market_data_dict: 市场数据字典（symbol -> MarketData）
            quantities: 持仓数量字典（contract_id -> quantity），默认每个合约1手

        Returns:
            组合定价结果
        """
        quantities = quantities or {}

        pricing_results = []
        total_value = 0.0
        total_greeks = GreeksResult()

        for contract in contracts:
            # 获取市场数据
            market_data = market_data_dict.get(contract.underlying_symbol)
            if not market_data:
                logger.warning(f"No market data for {contract.underlying_symbol}")
                continue

            # 定价
            if contract.is_european() and not contract.is_barrier():
                result = self.price_european_option(contract, market_data)
            elif contract.is_american():
                result = self.price_american_option(contract, market_data)
            elif contract.is_barrier():
                result = self.price_barrier_option(contract, market_data)
            else:
                result = self.price_european_option(contract, market_data)

            pricing_results.append(result)

            # 考虑持仓数量
            quantity = quantities.get(contract.contract_id, 1)
            contract_value = result.theoretical_price * quantity * contract.multiplier
            total_value += contract_value

            # 汇总Greeks
            scaled_greeks = result.greeks.scale(quantity * contract.multiplier)
            total_greeks = total_greeks.add(scaled_greeks)

        return PortfolioPricing(
            portfolio_id=str(uuid.uuid4())[:8],
            contracts=contracts,
            pricing_results=pricing_results,
            total_value=total_value,
            total_greeks=total_greeks
        )

    # ============ Volatility Surface ============

    def build_volatility_surface(
        self,
        underlying_symbol: str,
        market_prices: List[Tuple[OptionContract, float]],
        market_data: MarketData,
        surface_model: VolatilitySurfaceModel = VolatilitySurfaceModel.LINEAR_INTERPOLATION
    ) -> VolatilitySurface:
        """
        构建波动率曲面

        Args:
            underlying_symbol: 标的资产代码
            market_prices: (期权合约, 市场价格) 列表
            market_data: 市场数据
            surface_model: 曲面模型

        Returns:
            波动率曲面
        """
        points = []

        for contract, market_price in market_prices:
            # 计算隐含波动率
            iv, converged = self.calculate_implied_volatility(
                contract, market_price, market_data
            )

            if converged:
                point = VolatilityPoint(
                    strike=contract.strike,
                    expiry=contract.expiry,
                    implied_volatility=iv
                )
                points.append(point)

        surface = VolatilitySurface(
            underlying_symbol=underlying_symbol,
            points=points,
            surface_model=surface_model
        )

        # 缓存曲面
        self._volatility_surfaces[underlying_symbol] = surface

        return surface

    def get_volatility_surface(self, underlying_symbol: str) -> Optional[VolatilitySurface]:
        """获取波动率曲面"""
        return self._volatility_surfaces.get(underlying_symbol)

    # ============ Statistics ============

    def get_pricing_statistics(self) -> Dict[str, Any]:
        """
        获取定价统计信息

        Returns:
            统计信息字典
        """
        cache_hit_rate = 0.0
        if self._cache_hits + self._cache_misses > 0:
            cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)

        return {
            "state": self._state.value,
            "total_pricings": self._pricing_count,
            "cache_enabled": self.config.enable_cache,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "volatility_surfaces": len(self._volatility_surfaces),
            "config": self.config.to_dict(),
        }

    def clear_cache(self, reset_stats: bool = False) -> None:
        """
        清空缓存

        Args:
            reset_stats: 是否同时重置统计计数器
        """
        if self._cache:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Pricing cache cleared")

        if reset_stats:
            self._pricing_count = 0
            logger.info("Pricing statistics reset")

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._pricing_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("All statistics reset")


# ============ Factory Functions ============

def create_option_pricing_engine(
    pricing_model: str = "black_scholes",
    enable_cache: bool = True,
    **config_kwargs
) -> OptionPricingEngine:
    """
    创建期权定价引擎

    Args:
        pricing_model: 默认定价模型
        enable_cache: 启用缓存
        **config_kwargs: 其他配置参数

    Returns:
        期权定价引擎实例
    """
    config = PricingConfig(
        default_model=PricingModel(pricing_model),
        enable_cache=enable_cache,
        **{k: v for k, v in config_kwargs.items() if hasattr(PricingConfig, k)}
    )

    return OptionPricingEngine(config)


def create_option_contract(
    contract_id: str,
    underlying_symbol: str,
    option_symbol: str,
    option_type: str,
    option_style: str,
    strike: float,
    expiry: datetime,
    multiplier: float = 100.0
) -> OptionContract:
    """
    创建期权合约

    Args:
        contract_id: 合约ID
        underlying_symbol: 标的代码
        option_symbol: 期权代码
        option_type: 期权类型 ("call" or "put")
        option_style: 期权风格 ("european" or "american")
        strike: 行权价
        expiry: 到期日
        multiplier: 合约乘数

    Returns:
        期权合约实例
    """
    return OptionContract(
        contract_id=contract_id,
        underlying_symbol=underlying_symbol,
        option_symbol=option_symbol,
        option_type=OptionType(option_type),
        option_style=OptionStyle(option_style),
        strike=strike,
        expiry=expiry,
        multiplier=multiplier
    )


def create_market_data(
    symbol: str,
    spot_price: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> MarketData:
    """
    创建市场数据

    Args:
        symbol: 资产代码
        spot_price: 现货价格
        volatility: 波动率
        risk_free_rate: 无风险利率
        dividend_yield: 股息率

    Returns:
        市场数据实例
    """
    return MarketData(
        symbol=symbol,
        spot_price=spot_price,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )


# ============ Exports ============

__all__ = [
    # Enums
    "OptionType",
    "OptionStyle",
    "BarrierType",
    "PricingModel",
    "VolatilitySurfaceModel",
    "EngineState",
    "CacheStrategy",
    # Data Classes
    "OptionContract",
    "MarketData",
    "GreeksResult",
    "PricingResult",
    "VolatilityPoint",
    "VolatilitySurface",
    "PortfolioPricing",
    "PricingConfig",
    # Pricing Models
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloModel",
    "LSMModel",
    "BarrierModel",
    # Calculators
    "ImpliedVolatilityCalculator",
    # Cache
    "PricingCache",
    # Main Engine
    "OptionPricingEngine",
    # Factory Functions
    "create_option_pricing_engine",
    "create_option_contract",
    "create_market_data",
    # Utilities
    "norm_cdf",
    "norm_pdf",
    "norm_inv",
]
