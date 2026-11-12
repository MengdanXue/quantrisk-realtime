"""
交易成本分析模块 (Transaction Cost Analysis / TCA)

提供全面的交易成本分析功能:
- 执行质量分析 (Implementation Shortfall)
- 市场影响模型 (Market Impact)
- 滑点分析 (Slippage Analysis)
- 时机成本分析 (Timing Cost)
- 执行基准比较 (Benchmark Comparison)
- 成本归因分解 (Cost Attribution)
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class ExecutionBenchmark(str, Enum):
    """执行基准类型"""
    ARRIVAL_PRICE = "arrival_price"           # 到达价格
    VWAP = "vwap"                              # 成交量加权平均价格
    TWAP = "twap"                              # 时间加权平均价格
    CLOSE_PRICE = "close_price"               # 收盘价
    OPEN_PRICE = "open_price"                 # 开盘价
    INTERVAL_VWAP = "interval_vwap"           # 区间VWAP
    PREVIOUS_CLOSE = "previous_close"         # 前收盘价


class CostComponent(str, Enum):
    """成本组成"""
    COMMISSION = "commission"                  # 佣金
    SPREAD = "spread"                          # 买卖价差
    MARKET_IMPACT = "market_impact"           # 市场影响
    TIMING_COST = "timing_cost"               # 时机成本
    DELAY_COST = "delay_cost"                 # 延迟成本
    SLIPPAGE = "slippage"                     # 滑点


class ImpactModel(str, Enum):
    """市场影响模型"""
    LINEAR = "linear"                          # 线性模型
    SQUARE_ROOT = "square_root"               # 平方根模型
    POWER_LAW = "power_law"                   # 幂律模型
    ALMGREN_CHRISS = "almgren_chriss"         # Almgren-Chriss模型
    KYLE = "kyle"                              # Kyle模型


class TradeDirection(str, Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


# ==================== 数据类 ====================

@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str
    symbol: str
    direction: TradeDirection
    quantity: float
    executed_price: float
    executed_time: datetime

    # 市场数据
    arrival_price: float = 0.0           # 到达价格(决策时价格)
    decision_time: Optional[datetime] = None  # 决策时间
    vwap: float = 0.0                    # 当日VWAP
    twap: float = 0.0                    # 当日TWAP
    close_price: float = 0.0             # 收盘价
    open_price: float = 0.0              # 开盘价
    previous_close: float = 0.0          # 前收盘价
    bid_price: float = 0.0               # 买入价
    ask_price: float = 0.0               # 卖出价

    # 执行细节
    commission: float = 0.0              # 佣金
    exchange_fee: float = 0.0            # 交易所费用
    total_cost: float = 0.0              # 总成本

    # 市场环境
    daily_volume: float = 0.0            # 当日成交量
    volatility: float = 0.0              # 波动率
    average_spread: float = 0.0          # 平均价差

    # 元数据
    order_id: Optional[str] = None
    strategy: str = ""
    portfolio_id: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def spread_cost(self) -> float:
        """买卖价差成本"""
        if self.bid_price > 0 and self.ask_price > 0:
            mid_price = (self.bid_price + self.ask_price) / 2
            if self.direction == TradeDirection.BUY:
                return (self.executed_price - mid_price) / mid_price
            else:
                return (mid_price - self.executed_price) / mid_price
        return 0.0

    @property
    def participation_rate(self) -> float:
        """参与率(交易量/日成交量)"""
        if self.daily_volume > 0:
            return self.quantity / self.daily_volume
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'executed_price': self.executed_price,
            'executed_time': self.executed_time.isoformat(),
            'arrival_price': self.arrival_price,
            'decision_time': self.decision_time.isoformat() if self.decision_time else None,
            'vwap': self.vwap,
            'twap': self.twap,
            'close_price': self.close_price,
            'open_price': self.open_price,
            'commission': self.commission,
            'total_cost': self.total_cost,
            'daily_volume': self.daily_volume,
            'volatility': self.volatility,
            'spread_cost': self.spread_cost,
            'participation_rate': self.participation_rate,
            'strategy': self.strategy,
            'portfolio_id': self.portfolio_id,
        }


@dataclass
class TCAResult:
    """TCA分析结果"""
    trade_id: str
    symbol: str
    direction: TradeDirection
    quantity: float
    notional_value: float              # 名义价值

    # 基准对比
    benchmark: ExecutionBenchmark
    benchmark_price: float
    executed_price: float

    # 成本分解 (basis points)
    implementation_shortfall: float    # 执行差价 (bps)
    market_impact: float               # 市场影响 (bps)
    timing_cost: float                 # 时机成本 (bps)
    delay_cost: float                  # 延迟成本 (bps)
    spread_cost: float                 # 价差成本 (bps)
    commission_cost: float             # 佣金成本 (bps)
    total_cost: float                  # 总成本 (bps)

    # 金额表示
    cost_in_currency: float            # 总成本(货币金额)

    # 执行质量评分
    execution_score: float             # 执行得分 (0-100)

    # 分析时间
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'notional_value': self.notional_value,
            'benchmark': self.benchmark.value,
            'benchmark_price': self.benchmark_price,
            'executed_price': self.executed_price,
            'implementation_shortfall': round(self.implementation_shortfall, 2),
            'market_impact': round(self.market_impact, 2),
            'timing_cost': round(self.timing_cost, 2),
            'delay_cost': round(self.delay_cost, 2),
            'spread_cost': round(self.spread_cost, 2),
            'commission_cost': round(self.commission_cost, 2),
            'total_cost': round(self.total_cost, 2),
            'cost_in_currency': round(self.cost_in_currency, 2),
            'execution_score': round(self.execution_score, 1),
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class PortfolioTCAResult:
    """组合TCA结果"""
    portfolio_id: str
    period_start: datetime
    period_end: datetime

    # 汇总统计
    total_trades: int
    total_notional: float
    total_cost_bps: float              # 总成本 (bps)
    total_cost_currency: float         # 总成本 (货币)

    # 成本分解
    avg_implementation_shortfall: float
    avg_market_impact: float
    avg_timing_cost: float
    avg_spread_cost: float
    avg_commission_cost: float

    # 按维度分析
    by_symbol: Dict[str, Dict[str, float]]
    by_direction: Dict[str, Dict[str, float]]
    by_strategy: Dict[str, Dict[str, float]]
    by_time_of_day: Dict[str, Dict[str, float]]

    # 执行质量
    avg_execution_score: float
    best_execution_pct: float          # 优于VWAP的交易比例

    # 详细结果
    trade_results: List[TCAResult] = field(default_factory=list)

    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'portfolio_id': self.portfolio_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'total_notional': round(self.total_notional, 2),
            'total_cost_bps': round(self.total_cost_bps, 2),
            'total_cost_currency': round(self.total_cost_currency, 2),
            'avg_implementation_shortfall': round(self.avg_implementation_shortfall, 2),
            'avg_market_impact': round(self.avg_market_impact, 2),
            'avg_timing_cost': round(self.avg_timing_cost, 2),
            'avg_spread_cost': round(self.avg_spread_cost, 2),
            'avg_commission_cost': round(self.avg_commission_cost, 2),
            'avg_execution_score': round(self.avg_execution_score, 1),
            'best_execution_pct': round(self.best_execution_pct * 100, 1),
            'by_symbol': self.by_symbol,
            'by_direction': self.by_direction,
            'by_strategy': self.by_strategy,
            'by_time_of_day': self.by_time_of_day,
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class MarketImpactEstimate:
    """市场影响估计"""
    symbol: str
    direction: TradeDirection
    quantity: float

    # 预估影响
    estimated_impact_bps: float        # 预估市场影响 (bps)
    impact_confidence: float           # 置信度 (0-1)

    # 模型参数
    model_used: ImpactModel
    participation_rate: float
    volatility: float
    average_daily_volume: float

    # 区间估计
    impact_low: float                  # 最低估计
    impact_high: float                 # 最高估计

    estimated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'estimated_impact_bps': round(self.estimated_impact_bps, 2),
            'impact_confidence': round(self.impact_confidence, 2),
            'model_used': self.model_used.value,
            'participation_rate': round(self.participation_rate * 100, 2),
            'volatility': round(self.volatility * 100, 2),
            'average_daily_volume': self.average_daily_volume,
            'impact_low': round(self.impact_low, 2),
            'impact_high': round(self.impact_high, 2),
            'estimated_at': self.estimated_at.isoformat(),
        }


@dataclass
class TCAConfig:
    """TCA配置"""
    default_benchmark: ExecutionBenchmark = ExecutionBenchmark.ARRIVAL_PRICE
    impact_model: ImpactModel = ImpactModel.SQUARE_ROOT

    # 模型参数
    impact_coefficient: float = 0.1     # 市场影响系数
    volatility_scaling: float = 1.0     # 波动率缩放
    participation_exponent: float = 0.5 # 参与率指数 (平方根模型)

    # 成本参数
    default_commission_bps: float = 5.0  # 默认佣金 (bps)
    default_spread_bps: float = 2.0      # 默认价差 (bps)

    # 执行评分权重
    vs_vwap_weight: float = 0.4
    vs_arrival_weight: float = 0.4
    market_impact_weight: float = 0.2

    # 时间分析桶
    time_buckets: List[str] = field(default_factory=lambda: [
        '09:00-10:00', '10:00-11:00', '11:00-12:00',
        '13:00-14:00', '14:00-15:00', '15:00-16:00'
    ])


# ==================== 市场影响模型 ====================

class MarketImpactModel:
    """市场影响模型基类"""

    def estimate_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        direction: TradeDirection,
        **kwargs
    ) -> float:
        """
        估计市场影响

        Args:
            quantity: 交易数量
            daily_volume: 日均成交量
            volatility: 波动率 (年化)
            direction: 交易方向

        Returns:
            预估市场影响 (bps)
        """
        raise NotImplementedError


class LinearImpactModel(MarketImpactModel):
    """线性市场影响模型"""

    def __init__(self, coefficient: float = 0.1):
        self.coefficient = coefficient

    def estimate_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        direction: TradeDirection,
        **kwargs
    ) -> float:
        """线性模型: Impact = coefficient * participation_rate * volatility"""
        if daily_volume <= 0:
            return 0.0

        participation_rate = quantity / daily_volume
        daily_vol = volatility / math.sqrt(252)

        impact = self.coefficient * participation_rate * daily_vol * 10000
        return impact


class SquareRootImpactModel(MarketImpactModel):
    """平方根市场影响模型"""

    def __init__(self, coefficient: float = 0.1, exponent: float = 0.5):
        self.coefficient = coefficient
        self.exponent = exponent

    def estimate_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        direction: TradeDirection,
        **kwargs
    ) -> float:
        """平方根模型: Impact = coefficient * sqrt(participation_rate) * volatility"""
        if daily_volume <= 0:
            return 0.0

        participation_rate = quantity / daily_volume
        daily_vol = volatility / math.sqrt(252)

        impact = self.coefficient * (participation_rate ** self.exponent) * daily_vol * 10000
        return impact


class PowerLawImpactModel(MarketImpactModel):
    """幂律市场影响模型"""

    def __init__(
        self,
        coefficient: float = 0.1,
        volume_exponent: float = 0.6,
        volatility_exponent: float = 1.0
    ):
        self.coefficient = coefficient
        self.volume_exponent = volume_exponent
        self.volatility_exponent = volatility_exponent

    def estimate_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        direction: TradeDirection,
        **kwargs
    ) -> float:
        """幂律模型: Impact = coefficient * (Q/V)^a * sigma^b"""
        if daily_volume <= 0:
            return 0.0

        participation_rate = quantity / daily_volume
        daily_vol = volatility / math.sqrt(252)

        impact = (
            self.coefficient *
            (participation_rate ** self.volume_exponent) *
            (daily_vol ** self.volatility_exponent) *
            10000
        )
        return impact


class AlmgrenChrissModel(MarketImpactModel):
    """
    Almgren-Chriss 市场影响模型

    考虑临时影响和永久影响
    """

    def __init__(
        self,
        eta: float = 0.001,       # 临时影响参数
        gamma: float = 0.0001,    # 永久影响参数
        lambda_param: float = 2e-6  # 风险厌恶参数
    ):
        self.eta = eta
        self.gamma = gamma
        self.lambda_param = lambda_param

    def estimate_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        direction: TradeDirection,
        **kwargs
    ) -> float:
        """Almgren-Chriss 模型"""
        if daily_volume <= 0:
            return 0.0

        daily_vol = volatility / math.sqrt(252)

        # 临时影响 (temporary impact)
        trading_rate = quantity  # 假设一次性执行
        temp_impact = self.eta * trading_rate

        # 永久影响 (permanent impact)
        perm_impact = self.gamma * quantity

        # 总影响 (以价格为单位，转换为bps)
        total_impact = temp_impact + perm_impact

        # 简化转换: 假设基准价格为1
        impact_bps = total_impact * 10000 * daily_vol

        return impact_bps


# ==================== TCA分析器 ====================

class TransactionCostAnalyzer:
    """交易成本分析器"""

    def __init__(self, config: Optional[TCAConfig] = None):
        self.config = config or TCAConfig()
        self._impact_models = self._initialize_impact_models()
        self._trade_history: List[TradeRecord] = []
        self._analysis_cache: Dict[str, TCAResult] = {}

    def _initialize_impact_models(self) -> Dict[ImpactModel, MarketImpactModel]:
        """初始化市场影响模型"""
        return {
            ImpactModel.LINEAR: LinearImpactModel(self.config.impact_coefficient),
            ImpactModel.SQUARE_ROOT: SquareRootImpactModel(
                self.config.impact_coefficient,
                self.config.participation_exponent
            ),
            ImpactModel.POWER_LAW: PowerLawImpactModel(self.config.impact_coefficient),
            ImpactModel.ALMGREN_CHRISS: AlmgrenChrissModel(),
        }

    def add_trade(self, trade: TradeRecord) -> None:
        """添加交易记录"""
        self._trade_history.append(trade)

    def add_trades(self, trades: List[TradeRecord]) -> None:
        """批量添加交易记录"""
        self._trade_history.extend(trades)

    def clear_history(self) -> None:
        """清除历史记录"""
        self._trade_history.clear()
        self._analysis_cache.clear()

    async def analyze_trade(
        self,
        trade: TradeRecord,
        benchmark: Optional[ExecutionBenchmark] = None
    ) -> TCAResult:
        """
        分析单笔交易

        Args:
            trade: 交易记录
            benchmark: 执行基准

        Returns:
            TCA分析结果
        """
        benchmark = benchmark or self.config.default_benchmark

        # 获取基准价格
        benchmark_price = self._get_benchmark_price(trade, benchmark)

        # 计算执行差价 (Implementation Shortfall)
        is_bps = self._calculate_implementation_shortfall(
            trade, benchmark_price
        )

        # 计算市场影响
        impact_bps = self._calculate_market_impact(trade)

        # 计算时机成本
        timing_bps = self._calculate_timing_cost(trade, benchmark_price)

        # 计算延迟成本
        delay_bps = self._calculate_delay_cost(trade)

        # 计算价差成本
        spread_bps = trade.spread_cost * 10000 if trade.spread_cost else self.config.default_spread_bps

        # 计算佣金成本
        commission_bps = self._calculate_commission_bps(trade)

        # 总成本
        total_bps = is_bps + commission_bps

        # 计算货币金额
        notional = trade.quantity * trade.executed_price
        cost_currency = notional * total_bps / 10000

        # 计算执行得分
        execution_score = self._calculate_execution_score(
            trade, is_bps, impact_bps
        )

        result = TCAResult(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            direction=trade.direction,
            quantity=trade.quantity,
            notional_value=notional,
            benchmark=benchmark,
            benchmark_price=benchmark_price,
            executed_price=trade.executed_price,
            implementation_shortfall=is_bps,
            market_impact=impact_bps,
            timing_cost=timing_bps,
            delay_cost=delay_bps,
            spread_cost=spread_bps,
            commission_cost=commission_bps,
            total_cost=total_bps,
            cost_in_currency=cost_currency,
            execution_score=execution_score,
        )

        self._analysis_cache[trade.trade_id] = result
        return result

    async def analyze_trades(
        self,
        trades: List[TradeRecord],
        benchmark: Optional[ExecutionBenchmark] = None
    ) -> List[TCAResult]:
        """批量分析交易"""
        results = []
        for trade in trades:
            result = await self.analyze_trade(trade, benchmark)
            results.append(result)
        return results

    async def analyze_portfolio(
        self,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime,
        trades: Optional[List[TradeRecord]] = None
    ) -> PortfolioTCAResult:
        """
        分析组合层面的交易成本

        Args:
            portfolio_id: 组合ID
            start_date: 开始日期
            end_date: 结束日期
            trades: 交易列表 (如未提供则使用历史记录)

        Returns:
            组合TCA结果
        """
        # 筛选交易
        if trades is None:
            trades = [
                t for t in self._trade_history
                if t.portfolio_id == portfolio_id
                and start_date <= t.executed_time <= end_date
            ]

        if not trades:
            return self._empty_portfolio_result(portfolio_id, start_date, end_date)

        # 分析所有交易
        results = await self.analyze_trades(trades)

        # 计算汇总统计
        total_notional = sum(r.notional_value for r in results)

        # 加权平均成本
        weighted_costs = self._calculate_weighted_averages(results)

        # 按维度分组
        by_symbol = self._group_by_dimension(results, 'symbol')
        by_direction = self._group_by_dimension(results, 'direction')
        by_strategy = self._group_by_strategy(results, trades)
        by_time = self._group_by_time(results, trades)

        # 执行质量统计
        avg_score = np.mean([r.execution_score for r in results])
        best_exec_pct = sum(
            1 for r in results if r.implementation_shortfall <= 0
        ) / len(results)

        return PortfolioTCAResult(
            portfolio_id=portfolio_id,
            period_start=start_date,
            period_end=end_date,
            total_trades=len(results),
            total_notional=total_notional,
            total_cost_bps=weighted_costs['total'],
            total_cost_currency=sum(r.cost_in_currency for r in results),
            avg_implementation_shortfall=weighted_costs['implementation_shortfall'],
            avg_market_impact=weighted_costs['market_impact'],
            avg_timing_cost=weighted_costs['timing_cost'],
            avg_spread_cost=weighted_costs['spread_cost'],
            avg_commission_cost=weighted_costs['commission_cost'],
            by_symbol=by_symbol,
            by_direction=by_direction,
            by_strategy=by_strategy,
            by_time_of_day=by_time,
            avg_execution_score=avg_score,
            best_execution_pct=best_exec_pct,
            trade_results=results,
        )

    def estimate_market_impact(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        daily_volume: float,
        volatility: float,
        model: Optional[ImpactModel] = None
    ) -> MarketImpactEstimate:
        """
        预估市场影响

        Args:
            symbol: 交易品种
            direction: 交易方向
            quantity: 交易数量
            daily_volume: 日均成交量
            volatility: 波动率
            model: 市场影响模型

        Returns:
            市场影响估计
        """
        model = model or self.config.impact_model
        impact_model = self._impact_models.get(model, self._impact_models[ImpactModel.SQUARE_ROOT])

        # 基础估计
        base_impact = impact_model.estimate_impact(
            quantity=quantity,
            daily_volume=daily_volume,
            volatility=volatility,
            direction=direction,
        )

        # 计算置信区间
        participation_rate = quantity / daily_volume if daily_volume > 0 else 0

        # 置信度随参与率下降
        confidence = max(0.5, 1.0 - participation_rate * 2)

        # 区间估计 (±30% 范围)
        impact_low = base_impact * 0.7
        impact_high = base_impact * 1.3

        return MarketImpactEstimate(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            estimated_impact_bps=base_impact,
            impact_confidence=confidence,
            model_used=model,
            participation_rate=participation_rate,
            volatility=volatility,
            average_daily_volume=daily_volume,
            impact_low=impact_low,
            impact_high=impact_high,
        )

    def get_cost_breakdown(
        self,
        trade_id: str
    ) -> Optional[Dict[str, float]]:
        """获取成本分解"""
        result = self._analysis_cache.get(trade_id)
        if not result:
            return None

        return {
            'implementation_shortfall': result.implementation_shortfall,
            'market_impact': result.market_impact,
            'timing_cost': result.timing_cost,
            'delay_cost': result.delay_cost,
            'spread_cost': result.spread_cost,
            'commission_cost': result.commission_cost,
            'total_cost': result.total_cost,
        }

    def compare_benchmarks(
        self,
        trade: TradeRecord
    ) -> Dict[str, float]:
        """比较不同基准的执行效果"""
        results = {}

        for benchmark in ExecutionBenchmark:
            benchmark_price = self._get_benchmark_price(trade, benchmark)
            if benchmark_price > 0:
                is_bps = self._calculate_implementation_shortfall(
                    trade, benchmark_price
                )
                results[benchmark.value] = is_bps

        return results

    # ==================== 私有方法 ====================

    def _get_benchmark_price(
        self,
        trade: TradeRecord,
        benchmark: ExecutionBenchmark
    ) -> float:
        """获取基准价格"""
        price_map = {
            ExecutionBenchmark.ARRIVAL_PRICE: trade.arrival_price,
            ExecutionBenchmark.VWAP: trade.vwap,
            ExecutionBenchmark.TWAP: trade.twap,
            ExecutionBenchmark.CLOSE_PRICE: trade.close_price,
            ExecutionBenchmark.OPEN_PRICE: trade.open_price,
            ExecutionBenchmark.PREVIOUS_CLOSE: trade.previous_close,
            ExecutionBenchmark.INTERVAL_VWAP: trade.vwap,  # 简化处理
        }

        price = price_map.get(benchmark, 0.0)

        # 如果指定基准价格不可用，回退到执行价格
        if price <= 0:
            price = trade.executed_price

        return price

    def _calculate_implementation_shortfall(
        self,
        trade: TradeRecord,
        benchmark_price: float
    ) -> float:
        """
        计算执行差价 (Implementation Shortfall)

        IS = (Executed Price - Benchmark Price) / Benchmark Price * 10000
        对于买入，正值表示成本
        对于卖出，需要反转符号
        """
        if benchmark_price <= 0:
            return 0.0

        price_diff = trade.executed_price - benchmark_price
        is_pct = price_diff / benchmark_price

        # 对于卖出，价格差异的符号相反
        if trade.direction == TradeDirection.SELL:
            is_pct = -is_pct

        return is_pct * 10000

    def _calculate_market_impact(self, trade: TradeRecord) -> float:
        """计算市场影响"""
        if trade.daily_volume <= 0 or trade.volatility <= 0:
            return 0.0

        impact_model = self._impact_models.get(
            self.config.impact_model,
            self._impact_models[ImpactModel.SQUARE_ROOT]
        )

        return impact_model.estimate_impact(
            quantity=trade.quantity,
            daily_volume=trade.daily_volume,
            volatility=trade.volatility,
            direction=trade.direction,
        )

    def _calculate_timing_cost(
        self,
        trade: TradeRecord,
        benchmark_price: float
    ) -> float:
        """
        计算时机成本

        时机成本 = 到达价格与VWAP之间的差异
        """
        if trade.arrival_price <= 0 or trade.vwap <= 0:
            return 0.0

        # 到达价格到VWAP的移动
        timing_diff = trade.vwap - trade.arrival_price
        timing_pct = timing_diff / trade.arrival_price

        if trade.direction == TradeDirection.SELL:
            timing_pct = -timing_pct

        return timing_pct * 10000

    def _calculate_delay_cost(self, trade: TradeRecord) -> float:
        """
        计算延迟成本

        延迟成本 = 决策时间到开始执行时间之间的价格移动
        """
        if not trade.decision_time or trade.arrival_price <= 0:
            return 0.0

        # 简化：使用到达价格与前收盘价的差异作为延迟成本的近似
        if trade.previous_close <= 0:
            return 0.0

        delay_diff = trade.arrival_price - trade.previous_close
        delay_pct = delay_diff / trade.previous_close

        if trade.direction == TradeDirection.SELL:
            delay_pct = -delay_pct

        return delay_pct * 10000

    def _calculate_commission_bps(self, trade: TradeRecord) -> float:
        """计算佣金成本 (bps)"""
        if trade.commission > 0:
            notional = trade.quantity * trade.executed_price
            if notional > 0:
                return (trade.commission / notional) * 10000
        return self.config.default_commission_bps

    def _calculate_execution_score(
        self,
        trade: TradeRecord,
        is_bps: float,
        impact_bps: float
    ) -> float:
        """
        计算执行得分 (0-100)

        考虑:
        - 相对VWAP的执行效果
        - 相对到达价格的执行效果
        - 市场影响的控制
        """
        score = 100.0

        # 相对VWAP (权重40%)
        if trade.vwap > 0:
            vs_vwap = self._calculate_implementation_shortfall(
                trade, trade.vwap
            )
            # 每高于VWAP 1bps扣1分
            vwap_penalty = max(0, vs_vwap) * 0.5
            score -= vwap_penalty * self.config.vs_vwap_weight

        # 相对到达价格 (权重40%)
        if trade.arrival_price > 0:
            vs_arrival = self._calculate_implementation_shortfall(
                trade, trade.arrival_price
            )
            arrival_penalty = max(0, vs_arrival) * 0.5
            score -= arrival_penalty * self.config.vs_arrival_weight

        # 市场影响控制 (权重20%)
        # 参与率超过5%开始扣分
        if trade.participation_rate > 0.05:
            impact_penalty = (trade.participation_rate - 0.05) * 200
            score -= impact_penalty * self.config.market_impact_weight

        return max(0, min(100, score))

    def _calculate_weighted_averages(
        self,
        results: List[TCAResult]
    ) -> Dict[str, float]:
        """计算名义价值加权平均"""
        total_notional = sum(r.notional_value for r in results)

        if total_notional <= 0:
            return {
                'implementation_shortfall': 0,
                'market_impact': 0,
                'timing_cost': 0,
                'spread_cost': 0,
                'commission_cost': 0,
                'total': 0,
            }

        return {
            'implementation_shortfall': sum(
                r.implementation_shortfall * r.notional_value for r in results
            ) / total_notional,
            'market_impact': sum(
                r.market_impact * r.notional_value for r in results
            ) / total_notional,
            'timing_cost': sum(
                r.timing_cost * r.notional_value for r in results
            ) / total_notional,
            'spread_cost': sum(
                r.spread_cost * r.notional_value for r in results
            ) / total_notional,
            'commission_cost': sum(
                r.commission_cost * r.notional_value for r in results
            ) / total_notional,
            'total': sum(
                r.total_cost * r.notional_value for r in results
            ) / total_notional,
        }

    def _group_by_dimension(
        self,
        results: List[TCAResult],
        dimension: str
    ) -> Dict[str, Dict[str, float]]:
        """按维度分组统计"""
        groups: Dict[str, List[TCAResult]] = defaultdict(list)

        for r in results:
            if dimension == 'symbol':
                key = r.symbol
            elif dimension == 'direction':
                key = r.direction.value
            else:
                key = 'unknown'
            groups[key].append(r)

        result = {}
        for key, group_results in groups.items():
            total_notional = sum(r.notional_value for r in group_results)
            if total_notional > 0:
                result[key] = {
                    'trades': len(group_results),
                    'notional': total_notional,
                    'avg_cost_bps': sum(
                        r.total_cost * r.notional_value for r in group_results
                    ) / total_notional,
                    'avg_score': np.mean([r.execution_score for r in group_results]),
                }

        return result

    def _group_by_strategy(
        self,
        results: List[TCAResult],
        trades: List[TradeRecord]
    ) -> Dict[str, Dict[str, float]]:
        """按策略分组"""
        trade_map = {t.trade_id: t for t in trades}
        groups: Dict[str, List[TCAResult]] = defaultdict(list)

        for r in results:
            trade = trade_map.get(r.trade_id)
            strategy = trade.strategy if trade else 'unknown'
            groups[strategy].append(r)

        result = {}
        for strategy, group_results in groups.items():
            total_notional = sum(r.notional_value for r in group_results)
            if total_notional > 0:
                result[strategy] = {
                    'trades': len(group_results),
                    'notional': total_notional,
                    'avg_cost_bps': sum(
                        r.total_cost * r.notional_value for r in group_results
                    ) / total_notional,
                    'avg_score': np.mean([r.execution_score for r in group_results]),
                }

        return result

    def _group_by_time(
        self,
        results: List[TCAResult],
        trades: List[TradeRecord]
    ) -> Dict[str, Dict[str, float]]:
        """按时间段分组"""
        trade_map = {t.trade_id: t for t in trades}
        groups: Dict[str, List[TCAResult]] = defaultdict(list)

        for r in results:
            trade = trade_map.get(r.trade_id)
            if trade:
                hour = trade.executed_time.hour
                bucket = self._get_time_bucket(hour)
                groups[bucket].append(r)

        result = {}
        for bucket, group_results in groups.items():
            total_notional = sum(r.notional_value for r in group_results)
            if total_notional > 0:
                result[bucket] = {
                    'trades': len(group_results),
                    'notional': total_notional,
                    'avg_cost_bps': sum(
                        r.total_cost * r.notional_value for r in group_results
                    ) / total_notional,
                    'avg_score': np.mean([r.execution_score for r in group_results]),
                }

        return result

    def _get_time_bucket(self, hour: int) -> str:
        """获取时间桶"""
        for bucket in self.config.time_buckets:
            start, end = bucket.split('-')
            start_hour = int(start.split(':')[0])
            end_hour = int(end.split(':')[0])
            if start_hour <= hour < end_hour:
                return bucket
        return 'other'

    def _empty_portfolio_result(
        self,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> PortfolioTCAResult:
        """返回空的组合结果"""
        return PortfolioTCAResult(
            portfolio_id=portfolio_id,
            period_start=start_date,
            period_end=end_date,
            total_trades=0,
            total_notional=0,
            total_cost_bps=0,
            total_cost_currency=0,
            avg_implementation_shortfall=0,
            avg_market_impact=0,
            avg_timing_cost=0,
            avg_spread_cost=0,
            avg_commission_cost=0,
            by_symbol={},
            by_direction={},
            by_strategy={},
            by_time_of_day={},
            avg_execution_score=0,
            best_execution_pct=0,
        )


# ==================== 执行质量报告生成器 ====================

class ExecutionQualityReporter:
    """执行质量报告生成器"""

    def __init__(self, analyzer: TransactionCostAnalyzer):
        self.analyzer = analyzer

    def generate_summary_report(
        self,
        portfolio_result: PortfolioTCAResult
    ) -> Dict[str, Any]:
        """生成摘要报告"""
        return {
            'report_type': 'execution_quality_summary',
            'period': {
                'start': portfolio_result.period_start.isoformat(),
                'end': portfolio_result.period_end.isoformat(),
            },
            'overview': {
                'total_trades': portfolio_result.total_trades,
                'total_notional': portfolio_result.total_notional,
                'total_cost_bps': portfolio_result.total_cost_bps,
                'total_cost_currency': portfolio_result.total_cost_currency,
            },
            'cost_breakdown': {
                'implementation_shortfall': portfolio_result.avg_implementation_shortfall,
                'market_impact': portfolio_result.avg_market_impact,
                'timing_cost': portfolio_result.avg_timing_cost,
                'spread_cost': portfolio_result.avg_spread_cost,
                'commission_cost': portfolio_result.avg_commission_cost,
            },
            'execution_quality': {
                'avg_score': portfolio_result.avg_execution_score,
                'best_execution_rate': portfolio_result.best_execution_pct,
            },
            'analysis_by_dimension': {
                'by_symbol': portfolio_result.by_symbol,
                'by_direction': portfolio_result.by_direction,
                'by_strategy': portfolio_result.by_strategy,
                'by_time_of_day': portfolio_result.by_time_of_day,
            },
            'generated_at': datetime.now().isoformat(),
        }

    def generate_trade_detail_report(
        self,
        results: List[TCAResult]
    ) -> List[Dict[str, Any]]:
        """生成交易明细报告"""
        return [r.to_dict() for r in results]

    def generate_benchmark_comparison(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, Dict[str, float]]:
        """生成基准比较报告"""
        comparison = {}

        for trade in trades:
            comparison[trade.trade_id] = self.analyzer.compare_benchmarks(trade)

        # 计算各基准的平均表现
        benchmark_avgs = defaultdict(list)
        for trade_comparisons in comparison.values():
            for benchmark, value in trade_comparisons.items():
                benchmark_avgs[benchmark].append(value)

        return {
            'by_trade': comparison,
            'averages': {
                benchmark: np.mean(values)
                for benchmark, values in benchmark_avgs.items()
            },
        }


# ==================== 工厂函数 ====================

def create_tca_analyzer(
    default_benchmark: str = 'arrival_price',
    impact_model: str = 'square_root',
    **kwargs
) -> TransactionCostAnalyzer:
    """
    创建TCA分析器

    Args:
        default_benchmark: 默认基准类型
        impact_model: 市场影响模型
        **kwargs: 其他配置参数

    Returns:
        TransactionCostAnalyzer实例
    """
    config = TCAConfig(
        default_benchmark=ExecutionBenchmark(default_benchmark),
        impact_model=ImpactModel(impact_model),
        **{k: v for k, v in kwargs.items() if hasattr(TCAConfig, k)}
    )
    return TransactionCostAnalyzer(config)


def create_trade_record(
    trade_id: str,
    symbol: str,
    direction: str,
    quantity: float,
    executed_price: float,
    executed_time: Optional[datetime] = None,
    **kwargs
) -> TradeRecord:
    """
    创建交易记录

    Args:
        trade_id: 交易ID
        symbol: 交易品种
        direction: 方向 ('buy' 或 'sell')
        quantity: 数量
        executed_price: 执行价格
        executed_time: 执行时间
        **kwargs: 其他参数

    Returns:
        TradeRecord实例
    """
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        direction=TradeDirection(direction),
        quantity=quantity,
        executed_price=executed_price,
        executed_time=executed_time or datetime.now(),
        **{k: v for k, v in kwargs.items() if hasattr(TradeRecord, k)}
    )


async def quick_tca(
    trade: TradeRecord,
    benchmark: str = 'arrival_price'
) -> TCAResult:
    """
    快速TCA分析

    Args:
        trade: 交易记录
        benchmark: 基准类型

    Returns:
        TCA结果
    """
    analyzer = create_tca_analyzer(default_benchmark=benchmark)
    return await analyzer.analyze_trade(trade)


def estimate_impact(
    symbol: str,
    direction: str,
    quantity: float,
    daily_volume: float,
    volatility: float,
    model: str = 'square_root'
) -> MarketImpactEstimate:
    """
    快速市场影响估计

    Args:
        symbol: 品种
        direction: 方向
        quantity: 数量
        daily_volume: 日均成交量
        volatility: 波动率
        model: 影响模型

    Returns:
        市场影响估计
    """
    analyzer = create_tca_analyzer(impact_model=model)
    return analyzer.estimate_market_impact(
        symbol=symbol,
        direction=TradeDirection(direction),
        quantity=quantity,
        daily_volume=daily_volume,
        volatility=volatility,
        model=ImpactModel(model),
    )
