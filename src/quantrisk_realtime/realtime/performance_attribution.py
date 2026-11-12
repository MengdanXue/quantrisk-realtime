"""
绩效归因模块
Performance Attribution Module

提供:
- Brinson归因分析（配置效应、选择效应、交互效应）
- 风险归因分析（VaR贡献、波动率贡献）
- 因子归因分析（市场因子、行业因子、风格因子）
- 交易归因分析（择时、选股、执行成本）

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# ============ Enums ============

class AttributionMethod(Enum):
    """归因方法"""
    BRINSON = "brinson"           # Brinson归因
    FACTOR = "factor"             # 因子归因
    RISK = "risk"                 # 风险归因
    TRADING = "trading"           # 交易归因
    SECTOR = "sector"             # 行业归因


class AttributionPeriod(Enum):
    """归因周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class FactorType(Enum):
    """因子类型"""
    MARKET = "market"             # 市场因子
    SIZE = "size"                 # 规模因子
    VALUE = "value"               # 价值因子
    MOMENTUM = "momentum"         # 动量因子
    QUALITY = "quality"           # 质量因子
    VOLATILITY = "volatility"     # 波动率因子
    LIQUIDITY = "liquidity"       # 流动性因子
    SECTOR = "sector"             # 行业因子
    CUSTOM = "custom"             # 自定义因子


# ============ Data Classes ============

@dataclass
class PositionData:
    """持仓数据"""
    symbol: str
    weight: float                     # 权重
    return_value: float               # 收益
    sector: Optional[str] = None      # 行业
    factor_exposures: Dict[str, float] = field(default_factory=dict)  # 因子暴露


@dataclass
class BenchmarkData:
    """基准数据"""
    symbol: str
    weight: float
    return_value: float
    sector: Optional[str] = None
    factor_exposures: Dict[str, float] = field(default_factory=dict)


@dataclass
class BrinsonResult:
    """Brinson归因结果"""
    allocation_effect: float          # 配置效应
    selection_effect: float           # 选择效应
    interaction_effect: float         # 交互效应
    total_active_return: float        # 总超额收益
    sector_allocation: Dict[str, float] = field(default_factory=dict)  # 各行业配置效应
    sector_selection: Dict[str, float] = field(default_factory=dict)   # 各行业选择效应
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'allocation_effect': self.allocation_effect,
            'selection_effect': self.selection_effect,
            'interaction_effect': self.interaction_effect,
            'total_active_return': self.total_active_return,
            'sector_allocation': self.sector_allocation,
            'sector_selection': self.sector_selection,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class FactorResult:
    """因子归因结果"""
    factor_returns: Dict[str, float]      # 各因子收益贡献
    factor_exposures: Dict[str, float]    # 各因子暴露
    specific_return: float                # 特异性收益（残差）
    total_factor_return: float            # 总因子收益
    r_squared: float                      # R方（解释度）
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_returns': self.factor_returns,
            'factor_exposures': self.factor_exposures,
            'specific_return': self.specific_return,
            'total_factor_return': self.total_factor_return,
            'r_squared': self.r_squared,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class RiskResult:
    """风险归因结果"""
    var_contribution: Dict[str, float]        # 各资产VaR贡献
    volatility_contribution: Dict[str, float] # 各资产波动率贡献
    marginal_var: Dict[str, float]            # 边际VaR
    component_var: Dict[str, float]           # 成分VaR
    total_var: float                          # 总VaR
    total_volatility: float                   # 总波动率
    diversification_ratio: float              # 分散化比率
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'var_contribution': self.var_contribution,
            'volatility_contribution': self.volatility_contribution,
            'marginal_var': self.marginal_var,
            'component_var': self.component_var,
            'total_var': self.total_var,
            'total_volatility': self.total_volatility,
            'diversification_ratio': self.diversification_ratio,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TradingResult:
    """交易归因结果"""
    timing_effect: float              # 择时效应
    selection_effect: float           # 选股效应
    execution_cost: float             # 执行成本
    slippage_cost: float              # 滑点成本
    commission_cost: float            # 佣金成本
    market_impact: float              # 市场冲击成本
    net_alpha: float                  # 净Alpha
    gross_alpha: float                # 毛Alpha
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timing_effect': self.timing_effect,
            'selection_effect': self.selection_effect,
            'execution_cost': self.execution_cost,
            'slippage_cost': self.slippage_cost,
            'commission_cost': self.commission_cost,
            'market_impact': self.market_impact,
            'net_alpha': self.net_alpha,
            'gross_alpha': self.gross_alpha,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AttributionConfig:
    """归因配置"""
    method: AttributionMethod = AttributionMethod.BRINSON
    period: AttributionPeriod = AttributionPeriod.DAILY
    confidence_level: float = 0.95        # VaR置信水平
    risk_free_rate: float = 0.02          # 无风险利率
    factors: List[str] = field(default_factory=lambda: ['market', 'size', 'value', 'momentum'])
    use_geometric_linking: bool = True    # 使用几何链接法
    include_currency: bool = False        # 包含货币效应


@dataclass
class AttributionSummary:
    """归因摘要"""
    portfolio_return: float
    benchmark_return: float
    active_return: float
    brinson: Optional[BrinsonResult] = None
    factor: Optional[FactorResult] = None
    risk: Optional[RiskResult] = None
    trading: Optional[TradingResult] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'portfolio_return': self.portfolio_return,
            'benchmark_return': self.benchmark_return,
            'active_return': self.active_return,
            'brinson': self.brinson.to_dict() if self.brinson else None,
            'factor': self.factor.to_dict() if self.factor else None,
            'risk': self.risk.to_dict() if self.risk else None,
            'trading': self.trading.to_dict() if self.trading else None,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
        }


# ============ Brinson Attribution ============

class BrinsonAttributor:
    """
    Brinson归因分析器

    实现Brinson-Hood-Beebower (BHB)和Brinson-Fachler (BF)归因模型

    BHB模型:
    - 配置效应 = Σ(wp,i - wb,i) × rb,i
    - 选择效应 = Σwb,i × (rp,i - rb,i)
    - 交互效应 = Σ(wp,i - wb,i) × (rp,i - rb,i)

    其中:
    - wp,i = 组合在行业i的权重
    - wb,i = 基准在行业i的权重
    - rp,i = 组合在行业i的收益
    - rb,i = 基准在行业i的收益
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()

    def calculate(
        self,
        portfolio: List[PositionData],
        benchmark: List[BenchmarkData],
    ) -> BrinsonResult:
        """
        计算Brinson归因

        Args:
            portfolio: 组合持仓列表
            benchmark: 基准持仓列表

        Returns:
            BrinsonResult: 归因结果
        """
        # 按行业聚合
        portfolio_by_sector = self._aggregate_by_sector(portfolio)
        benchmark_by_sector = self._aggregate_by_sector(benchmark)

        # 获取所有行业
        all_sectors = set(portfolio_by_sector.keys()) | set(benchmark_by_sector.keys())

        # 计算各行业的归因效应
        sector_allocation = {}
        sector_selection = {}
        sector_interaction = {}

        for sector in all_sectors:
            p_weight = portfolio_by_sector.get(sector, (0.0, 0.0))[0]
            p_return = portfolio_by_sector.get(sector, (0.0, 0.0))[1]
            b_weight = benchmark_by_sector.get(sector, (0.0, 0.0))[0]
            b_return = benchmark_by_sector.get(sector, (0.0, 0.0))[1]

            # BHB模型
            # 配置效应：选择了不同于基准的行业权重
            allocation = (p_weight - b_weight) * b_return
            # 选择效应：在相同行业内选择了不同的证券
            selection = b_weight * (p_return - b_return)
            # 交互效应：配置和选择的联合效应
            interaction = (p_weight - b_weight) * (p_return - b_return)

            sector_allocation[sector] = allocation
            sector_selection[sector] = selection
            sector_interaction[sector] = interaction

        # 汇总
        total_allocation = sum(sector_allocation.values())
        total_selection = sum(sector_selection.values())
        total_interaction = sum(sector_interaction.values())
        total_active_return = total_allocation + total_selection + total_interaction

        return BrinsonResult(
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            total_active_return=total_active_return,
            sector_allocation=sector_allocation,
            sector_selection=sector_selection,
        )

    def _aggregate_by_sector(
        self,
        positions: List[Union[PositionData, BenchmarkData]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        按行业聚合持仓

        Returns:
            Dict[sector, (total_weight, weighted_return)]
        """
        sector_data: Dict[str, List[Tuple[float, float]]] = {}

        for pos in positions:
            sector = pos.sector or 'Other'
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append((pos.weight, pos.return_value))

        result = {}
        for sector, items in sector_data.items():
            total_weight = sum(w for w, _ in items)
            if total_weight > 0:
                weighted_return = sum(w * r for w, r in items) / total_weight
            else:
                weighted_return = 0.0
            result[sector] = (total_weight, weighted_return)

        return result


# ============ Factor Attribution ============

class FactorAttributor:
    """
    因子归因分析器

    基于多因子模型进行收益归因：
    R = α + β1*F1 + β2*F2 + ... + βn*Fn + ε

    其中:
    - R = 组合收益
    - α = Alpha（超额收益）
    - βi = 第i个因子的暴露
    - Fi = 第i个因子的收益
    - ε = 残差（特异性收益）
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()
        self.factor_returns: Dict[str, float] = {}  # 因子收益率

    def set_factor_returns(self, factor_returns: Dict[str, float]):
        """设置因子收益率"""
        self.factor_returns = factor_returns

    def calculate(
        self,
        portfolio: List[PositionData],
        portfolio_return: float,
    ) -> FactorResult:
        """
        计算因子归因

        Args:
            portfolio: 组合持仓列表
            portfolio_return: 组合总收益

        Returns:
            FactorResult: 归因结果
        """
        # 计算组合级别的因子暴露（加权平均）
        portfolio_exposures = self._calculate_portfolio_exposures(portfolio)

        # 计算各因子的收益贡献
        factor_contributions = {}
        total_factor_return = 0.0

        for factor, exposure in portfolio_exposures.items():
            factor_return = self.factor_returns.get(factor, 0.0)
            contribution = exposure * factor_return
            factor_contributions[factor] = contribution
            total_factor_return += contribution

        # 特异性收益（残差）
        specific_return = portfolio_return - total_factor_return

        # 计算R方（解释度）
        if portfolio_return != 0:
            r_squared = 1 - (specific_return ** 2) / (portfolio_return ** 2)
            r_squared = max(0.0, min(1.0, r_squared))  # 限制在0-1之间
        else:
            r_squared = 0.0

        return FactorResult(
            factor_returns=factor_contributions,
            factor_exposures=portfolio_exposures,
            specific_return=specific_return,
            total_factor_return=total_factor_return,
            r_squared=r_squared,
        )

    def _calculate_portfolio_exposures(
        self,
        portfolio: List[PositionData]
    ) -> Dict[str, float]:
        """计算组合级别的因子暴露"""
        total_weight = sum(pos.weight for pos in portfolio)
        if total_weight == 0:
            return {}

        exposures: Dict[str, float] = {}

        for pos in portfolio:
            normalized_weight = pos.weight / total_weight
            for factor, exposure in pos.factor_exposures.items():
                if factor not in exposures:
                    exposures[factor] = 0.0
                exposures[factor] += normalized_weight * exposure

        return exposures


# ============ Risk Attribution ============

class RiskAttributor:
    """
    风险归因分析器

    分析各资产/因子对组合风险的贡献:
    - VaR分解
    - 波动率分解
    - 边际风险贡献
    - 成分风险贡献
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()

    def calculate(
        self,
        weights: Dict[str, float],
        returns_matrix: np.ndarray,
        asset_names: List[str],
    ) -> RiskResult:
        """
        计算风险归因

        Args:
            weights: 资产权重字典
            returns_matrix: 收益率矩阵 (n_periods x n_assets)
            asset_names: 资产名称列表

        Returns:
            RiskResult: 风险归因结果
        """
        n_assets = len(asset_names)

        # 构建权重向量
        weight_vector = np.array([weights.get(name, 0.0) for name in asset_names])

        # 计算协方差矩阵
        if returns_matrix.shape[0] > 1:
            cov_matrix = np.cov(returns_matrix, rowvar=False)
            if cov_matrix.ndim == 0:  # 单资产情况
                cov_matrix = np.array([[cov_matrix]])
        else:
            cov_matrix = np.zeros((n_assets, n_assets))

        # 组合方差和波动率
        portfolio_variance = weight_vector @ cov_matrix @ weight_vector
        portfolio_volatility = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0.0

        # 边际VaR和边际波动率贡献
        marginal_var = {}
        marginal_vol = {}

        z_score = self._get_z_score(self.config.confidence_level)

        if portfolio_volatility > 0:
            marginal_contribution = cov_matrix @ weight_vector / portfolio_volatility
            for i, name in enumerate(asset_names):
                marginal_vol[name] = marginal_contribution[i]
                marginal_var[name] = z_score * marginal_contribution[i]
        else:
            for name in asset_names:
                marginal_vol[name] = 0.0
                marginal_var[name] = 0.0

        # 成分VaR和成分波动率贡献
        component_var = {}
        volatility_contribution = {}

        for i, name in enumerate(asset_names):
            component = weight_vector[i] * marginal_vol.get(name, 0.0)
            volatility_contribution[name] = component
            component_var[name] = z_score * component

        # VaR贡献（归一化）
        total_component_var = sum(component_var.values())
        var_contribution = {}
        if total_component_var != 0:
            for name in asset_names:
                var_contribution[name] = component_var[name] / total_component_var
        else:
            for name in asset_names:
                var_contribution[name] = 0.0

        # 总VaR
        total_var = z_score * portfolio_volatility

        # 分散化比率
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_sum_vols = sum(
            weight_vector[i] * individual_vols[i]
            for i in range(n_assets)
        )
        diversification_ratio = weighted_sum_vols / portfolio_volatility if portfolio_volatility > 0 else 1.0

        return RiskResult(
            var_contribution=var_contribution,
            volatility_contribution=volatility_contribution,
            marginal_var=marginal_var,
            component_var=component_var,
            total_var=total_var,
            total_volatility=portfolio_volatility,
            diversification_ratio=diversification_ratio,
        )

    def _get_z_score(self, confidence_level: float) -> float:
        """获取Z分数"""
        # 常用置信水平对应的Z分数
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
        }
        return z_scores.get(confidence_level, 1.645)


# ============ Trading Attribution ============

class TradingAttributor:
    """
    交易归因分析器

    分析交易活动对收益的影响:
    - 择时效应：进出时机的选择
    - 选股效应：标的选择的贡献
    - 执行成本：滑点、佣金、市场冲击
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()

    def calculate(
        self,
        trades: List[Dict[str, Any]],
        benchmark_returns: Dict[str, float],
    ) -> TradingResult:
        """
        计算交易归因

        Args:
            trades: 交易记录列表，每条记录包含:
                - symbol: 标的
                - side: 方向 (buy/sell)
                - quantity: 数量
                - price: 成交价
                - benchmark_price: 基准价格（如VWAP）
                - commission: 佣金
                - timestamp: 时间戳
            benchmark_returns: 基准收益

        Returns:
            TradingResult: 交易归因结果
        """
        if not trades:
            return TradingResult(
                timing_effect=0.0,
                selection_effect=0.0,
                execution_cost=0.0,
                slippage_cost=0.0,
                commission_cost=0.0,
                market_impact=0.0,
                net_alpha=0.0,
                gross_alpha=0.0,
            )

        total_value = 0.0
        timing_effect = 0.0
        selection_effect = 0.0
        slippage_cost = 0.0
        commission_cost = 0.0
        market_impact = 0.0

        for trade in trades:
            symbol = trade.get('symbol', '')
            side = trade.get('side', 'buy')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            benchmark_price = trade.get('benchmark_price', price)
            commission = trade.get('commission', 0)
            impact = trade.get('market_impact', 0)

            trade_value = quantity * price
            total_value += trade_value

            # 滑点：实际价格与基准价格的差异
            if side == 'buy':
                slippage = (price - benchmark_price) / benchmark_price if benchmark_price > 0 else 0
            else:
                slippage = (benchmark_price - price) / benchmark_price if benchmark_price > 0 else 0

            slippage_cost += slippage * trade_value
            commission_cost += commission
            market_impact += impact * trade_value if impact else 0

            # 择时效应：相对于持有基准的超额收益
            bench_return = benchmark_returns.get(symbol, 0.0)
            if side == 'buy':
                timing_effect += bench_return * trade_value
            else:
                timing_effect -= bench_return * trade_value

            # 选股效应：选择该标的带来的收益
            actual_return = trade.get('realized_return', 0.0)
            selection_effect += (actual_return - bench_return) * trade_value

        # 归一化
        if total_value > 0:
            timing_effect /= total_value
            selection_effect /= total_value
            slippage_cost /= total_value
            market_impact /= total_value

        # 总执行成本
        execution_cost = slippage_cost + commission_cost / total_value if total_value > 0 else 0
        execution_cost += market_impact

        # Alpha计算
        gross_alpha = timing_effect + selection_effect
        net_alpha = gross_alpha - execution_cost

        return TradingResult(
            timing_effect=timing_effect,
            selection_effect=selection_effect,
            execution_cost=execution_cost,
            slippage_cost=slippage_cost,
            commission_cost=commission_cost / total_value if total_value > 0 else 0,
            market_impact=market_impact,
            net_alpha=net_alpha,
            gross_alpha=gross_alpha,
        )


# ============ Main Performance Attribution Class ============

class PerformanceAttribution:
    """
    绩效归因主类

    整合多种归因方法，提供完整的绩效归因分析
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()
        self.brinson = BrinsonAttributor(config)
        self.factor = FactorAttributor(config)
        self.risk = RiskAttributor(config)
        self.trading = TradingAttributor(config)

        self._history: List[AttributionSummary] = []

    async def analyze(
        self,
        portfolio: List[PositionData],
        benchmark: List[BenchmarkData],
        portfolio_return: float,
        benchmark_return: float,
        returns_matrix: Optional[np.ndarray] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        factor_returns: Optional[Dict[str, float]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> AttributionSummary:
        """
        执行完整的绩效归因分析

        Args:
            portfolio: 组合持仓
            benchmark: 基准持仓
            portfolio_return: 组合收益
            benchmark_return: 基准收益
            returns_matrix: 历史收益矩阵（用于风险归因）
            trades: 交易记录（用于交易归因）
            factor_returns: 因子收益（用于因子归因）
            period_start: 分析期间开始
            period_end: 分析期间结束

        Returns:
            AttributionSummary: 归因摘要
        """
        active_return = portfolio_return - benchmark_return

        # Brinson归因
        brinson_result = None
        if self.config.method in [AttributionMethod.BRINSON, AttributionMethod.SECTOR]:
            try:
                brinson_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.brinson.calculate,
                    portfolio,
                    benchmark,
                )
            except Exception as e:
                logger.warning(f"Brinson attribution failed: {e}")

        # 因子归因
        factor_result = None
        if self.config.method == AttributionMethod.FACTOR and factor_returns:
            try:
                self.factor.set_factor_returns(factor_returns)
                factor_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.factor.calculate,
                    portfolio,
                    portfolio_return,
                )
            except Exception as e:
                logger.warning(f"Factor attribution failed: {e}")

        # 风险归因
        risk_result = None
        if self.config.method == AttributionMethod.RISK and returns_matrix is not None:
            try:
                weights = {pos.symbol: pos.weight for pos in portfolio}
                asset_names = [pos.symbol for pos in portfolio]
                risk_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.risk.calculate,
                    weights,
                    returns_matrix,
                    asset_names,
                )
            except Exception as e:
                logger.warning(f"Risk attribution failed: {e}")

        # 交易归因
        trading_result = None
        if self.config.method == AttributionMethod.TRADING and trades:
            try:
                benchmark_returns = {b.symbol: b.return_value for b in benchmark}
                trading_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.trading.calculate,
                    trades,
                    benchmark_returns,
                )
            except Exception as e:
                logger.warning(f"Trading attribution failed: {e}")

        # 创建摘要
        summary = AttributionSummary(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            brinson=brinson_result,
            factor=factor_result,
            risk=risk_result,
            trading=trading_result,
            period_start=period_start,
            period_end=period_end,
        )

        # 保存历史
        self._history.append(summary)

        return summary

    def analyze_sync(
        self,
        portfolio: List[PositionData],
        benchmark: List[BenchmarkData],
        portfolio_return: float,
        benchmark_return: float,
        **kwargs
    ) -> AttributionSummary:
        """同步版本的分析方法"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.analyze(
                    portfolio=portfolio,
                    benchmark=benchmark,
                    portfolio_return=portfolio_return,
                    benchmark_return=benchmark_return,
                    **kwargs
                )
            )
        finally:
            loop.close()

    def get_history(self) -> List[AttributionSummary]:
        """获取历史归因记录"""
        return self._history.copy()

    def clear_history(self):
        """清除历史记录"""
        self._history.clear()

    async def analyze_multi_period(
        self,
        periods: List[Dict[str, Any]],
    ) -> List[AttributionSummary]:
        """
        多期间归因分析

        Args:
            periods: 各期间数据列表

        Returns:
            各期间归因结果
        """
        results = []

        for period in periods:
            summary = await self.analyze(**period)
            results.append(summary)

        return results

    def link_periods(
        self,
        summaries: List[AttributionSummary],
        method: str = 'geometric'
    ) -> AttributionSummary:
        """
        链接多期间归因结果

        Args:
            summaries: 各期间归因结果
            method: 链接方法 (geometric/arithmetic)

        Returns:
            链接后的总归因结果
        """
        if not summaries:
            return AttributionSummary(
                portfolio_return=0.0,
                benchmark_return=0.0,
                active_return=0.0,
            )

        if method == 'geometric':
            # 几何链接
            portfolio_return = np.prod([1 + s.portfolio_return for s in summaries]) - 1
            benchmark_return = np.prod([1 + s.benchmark_return for s in summaries]) - 1
        else:
            # 算术链接
            portfolio_return = sum(s.portfolio_return for s in summaries)
            benchmark_return = sum(s.benchmark_return for s in summaries)

        active_return = portfolio_return - benchmark_return

        # 链接Brinson结果（算术平均）
        brinson = None
        brinson_results = [s.brinson for s in summaries if s.brinson]
        if brinson_results:
            brinson = BrinsonResult(
                allocation_effect=sum(b.allocation_effect for b in brinson_results),
                selection_effect=sum(b.selection_effect for b in brinson_results),
                interaction_effect=sum(b.interaction_effect for b in brinson_results),
                total_active_return=sum(b.total_active_return for b in brinson_results),
            )

        return AttributionSummary(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            brinson=brinson,
            period_start=summaries[0].period_start if summaries else None,
            period_end=summaries[-1].period_end if summaries else None,
        )


# ============ Factory Functions ============

def create_attribution(
    method: str = 'brinson',
    period: str = 'daily',
    confidence_level: float = 0.95,
    **kwargs
) -> PerformanceAttribution:
    """
    创建绩效归因分析器

    Args:
        method: 归因方法 (brinson/factor/risk/trading)
        period: 分析周期
        confidence_level: VaR置信水平

    Returns:
        PerformanceAttribution实例
    """
    method_map = {
        'brinson': AttributionMethod.BRINSON,
        'factor': AttributionMethod.FACTOR,
        'risk': AttributionMethod.RISK,
        'trading': AttributionMethod.TRADING,
        'sector': AttributionMethod.SECTOR,
    }

    period_map = {
        'daily': AttributionPeriod.DAILY,
        'weekly': AttributionPeriod.WEEKLY,
        'monthly': AttributionPeriod.MONTHLY,
        'quarterly': AttributionPeriod.QUARTERLY,
        'yearly': AttributionPeriod.YEARLY,
    }

    config = AttributionConfig(
        method=method_map.get(method, AttributionMethod.BRINSON),
        period=period_map.get(period, AttributionPeriod.DAILY),
        confidence_level=confidence_level,
        **kwargs
    )

    return PerformanceAttribution(config)


def create_position(
    symbol: str,
    weight: float,
    return_value: float,
    sector: Optional[str] = None,
    **factor_exposures
) -> PositionData:
    """创建持仓数据"""
    return PositionData(
        symbol=symbol,
        weight=weight,
        return_value=return_value,
        sector=sector,
        factor_exposures=factor_exposures,
    )


def create_benchmark(
    symbol: str,
    weight: float,
    return_value: float,
    sector: Optional[str] = None,
    **factor_exposures
) -> BenchmarkData:
    """创建基准数据"""
    return BenchmarkData(
        symbol=symbol,
        weight=weight,
        return_value=return_value,
        sector=sector,
        factor_exposures=factor_exposures,
    )


async def quick_attribution(
    portfolio: List[Dict[str, Any]],
    benchmark: List[Dict[str, Any]],
    method: str = 'brinson',
) -> AttributionSummary:
    """
    快速归因分析

    Args:
        portfolio: 组合数据 [{'symbol': 'AAPL', 'weight': 0.1, 'return': 0.05, 'sector': 'Tech'}, ...]
        benchmark: 基准数据
        method: 归因方法

    Returns:
        归因摘要
    """
    # 转换数据
    portfolio_positions = [
        PositionData(
            symbol=p['symbol'],
            weight=p['weight'],
            return_value=p.get('return', p.get('return_value', 0)),
            sector=p.get('sector'),
            factor_exposures=p.get('factor_exposures', {}),
        )
        for p in portfolio
    ]

    benchmark_positions = [
        BenchmarkData(
            symbol=b['symbol'],
            weight=b['weight'],
            return_value=b.get('return', b.get('return_value', 0)),
            sector=b.get('sector'),
            factor_exposures=b.get('factor_exposures', {}),
        )
        for b in benchmark
    ]

    # 计算总收益
    portfolio_return = sum(p.weight * p.return_value for p in portfolio_positions)
    benchmark_return = sum(b.weight * b.return_value for b in benchmark_positions)

    # 执行归因
    attribution = create_attribution(method=method)
    return await attribution.analyze(
        portfolio=portfolio_positions,
        benchmark=benchmark_positions,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
    )


__all__ = [
    # Enums
    'AttributionMethod',
    'AttributionPeriod',
    'FactorType',
    # Data Classes
    'PositionData',
    'BenchmarkData',
    'BrinsonResult',
    'FactorResult',
    'RiskResult',
    'TradingResult',
    'AttributionConfig',
    'AttributionSummary',
    # Attributors
    'BrinsonAttributor',
    'FactorAttributor',
    'RiskAttributor',
    'TradingAttributor',
    'PerformanceAttribution',
    # Factory Functions
    'create_attribution',
    'create_position',
    'create_benchmark',
    'quick_attribution',
]
