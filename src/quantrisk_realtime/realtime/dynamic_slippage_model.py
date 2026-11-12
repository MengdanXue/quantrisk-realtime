"""
动态滑点模型

基于平方根市场冲击模型估算订单滑点，支持在线学习优化系数。

核心特性:
- 平方根冲击模型: impact = σ × √(Q/ADV) × coefficient
- 时段调整: 开盘(1.8x), 午休(1.3x), 收盘(1.5x)
- 流动性分级: 高/中/低流动性自适应
- 在线系数学习: 根据实际成交更新模型参数
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class LiquidityRegime(Enum):
    """流动性分级"""
    HIGH = "high"       # 日成交量 > ADV的120%
    NORMAL = "normal"   # 日成交量在80%-120%之间
    LOW = "low"         # 日成交量 < ADV的80%


@dataclass
class MarketDepth:
    """市场深度数据"""
    bid_prices: List[float]
    bid_volumes: List[float]
    ask_prices: List[float]
    ask_volumes: List[float]
    spread: float = 0.0

    def __post_init__(self):
        if self.bid_prices and self.ask_prices:
            self.spread = self.ask_prices[0] - self.bid_prices[0]

    @property
    def mid_price(self) -> float:
        """中间价"""
        if self.bid_prices and self.ask_prices:
            return (self.bid_prices[0] + self.ask_prices[0]) / 2
        return 0.0

    @property
    def total_bid_volume(self) -> float:
        """总买量"""
        return sum(self.bid_volumes) if self.bid_volumes else 0.0

    @property
    def total_ask_volume(self) -> float:
        """总卖量"""
        return sum(self.ask_volumes) if self.ask_volumes else 0.0


@dataclass
class SlippageBreakdown:
    """滑点分解"""
    spread_cost_bps: float      # 买卖价差成本 (bps)
    market_impact_bps: float    # 市场冲击成本 (bps)
    timing_adjustment_bps: float  # 时段调整 (bps)
    liquidity_adjustment_bps: float  # 流动性调整 (bps)

    @property
    def total_bps(self) -> float:
        """总滑点 (bps)"""
        return (self.spread_cost_bps + self.market_impact_bps +
                self.timing_adjustment_bps + self.liquidity_adjustment_bps)


@dataclass
class OrderSlice:
    """订单切片"""
    quantity: float
    delay_seconds: int
    expected_impact_bps: float


@dataclass
class SlippageEstimate:
    """滑点估算结果"""
    symbol: str
    order_side: OrderSide
    quantity: float
    expected_bps: float
    confidence_interval: Tuple[float, float]
    optimal_split_count: int
    liquidity_regime: LiquidityRegime
    breakdown: SlippageBreakdown
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SlippageModelConfig:
    """滑点模型配置"""
    # 平方根冲击模型基础系数
    base_coefficient: float = 0.1

    # 时段调整系数
    opening_multiplier: float = 1.8      # 开盘30分钟
    lunch_multiplier: float = 1.3        # 午休期间
    closing_multiplier: float = 1.5      # 收盘30分钟

    # 流动性调整系数
    low_liquidity_multiplier: float = 1.5
    high_liquidity_multiplier: float = 0.8

    # 订单切分参数
    max_volume_participation: float = 0.1  # 最大成交量参与率
    min_slice_quantity: float = 100.0       # 最小切片数量

    # 在线学习参数
    learning_rate: float = 0.05
    coefficient_decay: float = 0.995

    # 置信区间参数
    confidence_level: float = 0.95


class DynamicSlippageModel:
    """动态滑点模型"""

    def __init__(self, config: Optional[SlippageModelConfig] = None):
        self.config = config or SlippageModelConfig()
        # 每个标的的滑点系数
        self._coefficients: Dict[str, float] = {}
        # 预测vs实际记录
        self._prediction_history: Dict[str, List[Tuple[float, float]]] = {}
        # 标准差估计
        self._std_estimates: Dict[str, float] = {}

    def estimate_slippage(
        self,
        symbol: str,
        order_quantity: float,
        order_side: OrderSide,
        market_depth: MarketDepth,
        volatility: float,
        daily_volume: float,
        current_time: Optional[datetime] = None
    ) -> SlippageEstimate:
        """
        估算订单滑点

        Args:
            symbol: 标的代码
            order_quantity: 订单数量
            order_side: 订单方向
            market_depth: 市场深度
            volatility: 波动率 (年化)
            daily_volume: 日均成交量 (ADV)
            current_time: 当前时间 (可选)

        Returns:
            SlippageEstimate: 滑点估算结果
        """
        current_time = current_time or datetime.now()

        # 1. 计算买卖价差成本
        spread_cost_bps = self._calculate_spread_cost(market_depth, order_side)

        # 2. 计算市场冲击成本 (平方根模型)
        coefficient = self._get_coefficient(symbol)
        volume_ratio = order_quantity / daily_volume if daily_volume > 0 else 0.1
        daily_volatility = volatility / math.sqrt(252)

        # 平方根冲击: impact = σ × √(Q/ADV) × coefficient
        market_impact_bps = (
            daily_volatility *
            math.sqrt(volume_ratio) *
            coefficient *
            10000  # 转换为bps
        )

        # 3. 时段调整
        timing_multiplier = self._get_timing_multiplier(current_time)
        timing_adjustment_bps = market_impact_bps * (timing_multiplier - 1)

        # 4. 流动性调整
        liquidity_regime = self._assess_liquidity(market_depth, daily_volume)
        liquidity_multiplier = self._get_liquidity_multiplier(liquidity_regime)
        liquidity_adjustment_bps = market_impact_bps * (liquidity_multiplier - 1)

        # 5. 构建滑点分解
        breakdown = SlippageBreakdown(
            spread_cost_bps=spread_cost_bps,
            market_impact_bps=market_impact_bps,
            timing_adjustment_bps=timing_adjustment_bps,
            liquidity_adjustment_bps=liquidity_adjustment_bps
        )

        # 6. 计算总滑点和置信区间
        expected_bps = breakdown.total_bps
        confidence_interval = self._calculate_confidence_interval(
            symbol, expected_bps
        )

        # 7. 计算最优切分数量
        optimal_split = self._calculate_optimal_split(
            order_quantity, daily_volume
        )

        return SlippageEstimate(
            symbol=symbol,
            order_side=order_side,
            quantity=order_quantity,
            expected_bps=expected_bps,
            confidence_interval=confidence_interval,
            optimal_split_count=optimal_split,
            liquidity_regime=liquidity_regime,
            breakdown=breakdown
        )

    def update_coefficient(
        self,
        symbol: str,
        predicted_slippage: float,
        actual_slippage: float
    ) -> None:
        """
        根据实际成交更新滑点系数

        Args:
            symbol: 标的代码
            predicted_slippage: 预测滑点 (bps)
            actual_slippage: 实际滑点 (bps)
        """
        # 记录历史
        if symbol not in self._prediction_history:
            self._prediction_history[symbol] = []
        self._prediction_history[symbol].append((predicted_slippage, actual_slippage))

        # 限制历史长度
        if len(self._prediction_history[symbol]) > 100:
            self._prediction_history[symbol] = self._prediction_history[symbol][-100:]

        # 计算误差
        error = actual_slippage - predicted_slippage

        # 在线更新系数
        current_coef = self._get_coefficient(symbol)
        adjustment = self.config.learning_rate * error / (predicted_slippage + 1e-6)
        new_coef = current_coef * (1 + adjustment)

        # 应用衰减并限制范围
        new_coef = new_coef * self.config.coefficient_decay
        new_coef = max(0.01, min(1.0, new_coef))  # 限制在合理范围

        self._coefficients[symbol] = new_coef

        # 更新标准差估计
        self._update_std_estimate(symbol)

    def get_optimal_split(
        self,
        total_quantity: float,
        daily_volume: float,
        urgency: float = 0.5
    ) -> List[OrderSlice]:
        """
        获取最优订单切分方案

        Args:
            total_quantity: 总订单量
            daily_volume: 日均成交量
            urgency: 紧急程度 (0-1, 1最紧急)

        Returns:
            List[OrderSlice]: 订单切片列表
        """
        # 计算最优切片数
        participation_rate = self.config.max_volume_participation * (1 - 0.5 * urgency)
        slice_volume = daily_volume * participation_rate / 78  # 假设每分钟交易
        slice_count = max(1, int(total_quantity / slice_volume))

        # 确保不超过合理数量
        slice_count = min(slice_count, 20)

        slices = []
        remaining = total_quantity
        base_delay = int(60 * (1 - urgency))  # 基础延迟秒数

        for i in range(slice_count):
            slice_qty = remaining / (slice_count - i)
            slice_qty = max(self.config.min_slice_quantity, slice_qty)

            if slice_qty > remaining:
                slice_qty = remaining

            # 估算该切片的市场冲击
            volume_ratio = slice_qty / daily_volume
            impact_bps = self.config.base_coefficient * math.sqrt(volume_ratio) * 10000

            slices.append(OrderSlice(
                quantity=slice_qty,
                delay_seconds=base_delay * i,
                expected_impact_bps=impact_bps
            ))

            remaining -= slice_qty
            if remaining <= 0:
                break

        return slices

    def _get_coefficient(self, symbol: str) -> float:
        """获取标的的滑点系数"""
        return self._coefficients.get(symbol, self.config.base_coefficient)

    def _calculate_spread_cost(
        self,
        market_depth: MarketDepth,
        order_side: OrderSide
    ) -> float:
        """计算买卖价差成本 (bps)"""
        if market_depth.mid_price <= 0:
            return 0.0

        half_spread = market_depth.spread / 2
        spread_cost = half_spread / market_depth.mid_price * 10000

        return spread_cost

    def _get_timing_multiplier(self, current_time: datetime) -> float:
        """获取时段调整系数"""
        t = current_time.time()

        # 开盘30分钟 (9:30-10:00)
        if time(9, 30) <= t <= time(10, 0):
            return self.config.opening_multiplier

        # 午休期间 (11:30-13:00)
        if time(11, 30) <= t <= time(13, 0):
            return self.config.lunch_multiplier

        # 收盘30分钟 (14:30-15:00)
        if time(14, 30) <= t <= time(15, 0):
            return self.config.closing_multiplier

        return 1.0

    def _assess_liquidity(
        self,
        market_depth: MarketDepth,
        daily_volume: float
    ) -> LiquidityRegime:
        """评估流动性等级"""
        # 基于盘口深度与日均成交量的比例
        depth_volume = market_depth.total_bid_volume + market_depth.total_ask_volume

        if daily_volume <= 0:
            return LiquidityRegime.LOW

        depth_ratio = depth_volume / daily_volume

        if depth_ratio > 0.02:  # 盘口深度超过日均量的2%
            return LiquidityRegime.HIGH
        elif depth_ratio < 0.005:  # 盘口深度低于日均量的0.5%
            return LiquidityRegime.LOW
        else:
            return LiquidityRegime.NORMAL

    def _get_liquidity_multiplier(self, regime: LiquidityRegime) -> float:
        """获取流动性调整系数"""
        if regime == LiquidityRegime.HIGH:
            return self.config.high_liquidity_multiplier
        elif regime == LiquidityRegime.LOW:
            return self.config.low_liquidity_multiplier
        return 1.0

    def _calculate_confidence_interval(
        self,
        symbol: str,
        expected_bps: float
    ) -> Tuple[float, float]:
        """计算置信区间"""
        std = self._std_estimates.get(symbol, expected_bps * 0.3)
        z = 1.96 if self.config.confidence_level == 0.95 else 2.58

        lower = max(0, expected_bps - z * std)
        upper = expected_bps + z * std

        return (lower, upper)

    def _calculate_optimal_split(
        self,
        order_quantity: float,
        daily_volume: float
    ) -> int:
        """计算最优切分数量"""
        if daily_volume <= 0:
            return 1

        volume_ratio = order_quantity / daily_volume

        # 基于Almgren-Chriss最优执行理论的简化版本
        if volume_ratio < 0.01:
            return 1
        elif volume_ratio < 0.05:
            return 3
        elif volume_ratio < 0.1:
            return 5
        elif volume_ratio < 0.2:
            return 8
        else:
            return 10

    def _update_std_estimate(self, symbol: str) -> None:
        """更新标准差估计"""
        history = self._prediction_history.get(symbol, [])
        if len(history) < 5:
            return

        errors = [abs(p - a) for p, a in history[-20:]]
        self._std_estimates[symbol] = np.std(errors)

    def get_model_stats(self, symbol: str) -> Dict:
        """获取模型统计信息"""
        history = self._prediction_history.get(symbol, [])

        if not history:
            return {
                "symbol": symbol,
                "coefficient": self._get_coefficient(symbol),
                "samples": 0,
                "mae": None,
                "mpe": None
            }

        predictions, actuals = zip(*history)
        errors = [abs(p - a) for p, a in history]
        percentage_errors = [
            (a - p) / (p + 1e-6) * 100 for p, a in history
        ]

        return {
            "symbol": symbol,
            "coefficient": self._get_coefficient(symbol),
            "samples": len(history),
            "mae": np.mean(errors),
            "mpe": np.mean(percentage_errors),
            "std": self._std_estimates.get(symbol),
            "last_10_accuracy": np.mean([
                1 - min(abs(p - a) / (a + 1e-6), 1)
                for p, a in history[-10:]
            ]) * 100
        }


# 单例实例
_slippage_model_instance: Optional[DynamicSlippageModel] = None


def get_slippage_model() -> DynamicSlippageModel:
    """获取滑点模型单例"""
    global _slippage_model_instance
    if _slippage_model_instance is None:
        _slippage_model_instance = DynamicSlippageModel()
    return _slippage_model_instance