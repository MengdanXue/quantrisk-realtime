"""
组合风险监控器 (Portfolio Risk Monitor)

提供实时组合风险计算和监控功能:
- VaR (Value at Risk) 计算 (历史模拟、参数法、蒙特卡洛)
- CVaR (Conditional VaR / Expected Shortfall) 计算
- 压力测试 (Stress Testing)
- 情景分析 (Scenario Analysis)
- 风险因子分解
- 实时风险限额监控
- 风险预警和告警

Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
import math
import statistics

logger = logging.getLogger(__name__)


# ============== Enums ==============

class VaRMethod(str, Enum):
    """VaR计算方法"""
    HISTORICAL = "historical"           # 历史模拟法
    PARAMETRIC = "parametric"           # 参数法(方差-协方差)
    MONTE_CARLO = "monte_carlo"         # 蒙特卡洛模拟
    CORNISH_FISHER = "cornish_fisher"   # Cornish-Fisher扩展


class StressScenario(str, Enum):
    """预设压力场景"""
    MARKET_CRASH = "market_crash"               # 市场崩盘
    INTEREST_RATE_SHOCK = "rate_shock"          # 利率冲击
    CURRENCY_CRISIS = "currency_crisis"         # 汇率危机
    LIQUIDITY_CRISIS = "liquidity_crisis"       # 流动性危机
    VOLATILITY_SPIKE = "volatility_spike"       # 波动率飙升
    SECTOR_ROTATION = "sector_rotation"         # 行业轮动
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # 相关性崩溃
    CUSTOM = "custom"                           # 自定义场景


class RiskLimitType(str, Enum):
    """风险限额类型"""
    VAR_LIMIT = "var_limit"                     # VaR限额
    CVAR_LIMIT = "cvar_limit"                   # CVaR限额
    POSITION_LIMIT = "position_limit"           # 持仓限额
    CONCENTRATION_LIMIT = "concentration_limit" # 集中度限额
    DRAWDOWN_LIMIT = "drawdown_limit"           # 回撤限额
    EXPOSURE_LIMIT = "exposure_limit"           # 敞口限额
    LEVERAGE_LIMIT = "leverage_limit"           # 杠杆限额
    BETA_LIMIT = "beta_limit"                   # Beta限额
    VOLATILITY_LIMIT = "volatility_limit"       # 波动率限额


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitorState(str, Enum):
    """监控状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============== Data Classes ==============

@dataclass
class PositionRisk:
    """持仓风险数据"""
    symbol: str
    quantity: float
    market_value: float
    weight: float
    volatility: float
    var_contribution: float = 0.0
    cvar_contribution: float = 0.0
    beta: float = 1.0
    correlation: float = 0.0
    marginal_var: float = 0.0
    component_var: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'market_value': self.market_value,
            'weight': self.weight,
            'volatility': self.volatility,
            'var_contribution': self.var_contribution,
            'cvar_contribution': self.cvar_contribution,
            'beta': self.beta,
            'correlation': self.correlation,
            'marginal_var': self.marginal_var,
            'component_var': self.component_var,
        }


@dataclass
class VaRResult:
    """VaR计算结果"""
    var_value: float                    # VaR值
    confidence_level: float             # 置信度
    holding_period: int                 # 持有期(天)
    method: VaRMethod                   # 计算方法
    timestamp: datetime
    portfolio_value: float = 0.0
    var_pct: float = 0.0                # VaR百分比
    position_vars: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'var_value': self.var_value,
            'confidence_level': self.confidence_level,
            'holding_period': self.holding_period,
            'method': self.method.value,
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'var_pct': self.var_pct,
            'position_vars': self.position_vars,
        }


@dataclass
class CVaRResult:
    """CVaR计算结果"""
    cvar_value: float                   # CVaR值(Expected Shortfall)
    var_value: float                    # 对应的VaR值
    confidence_level: float
    holding_period: int
    method: VaRMethod
    timestamp: datetime
    portfolio_value: float = 0.0
    cvar_pct: float = 0.0
    tail_loss_count: int = 0            # 尾部损失样本数

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cvar_value': self.cvar_value,
            'var_value': self.var_value,
            'confidence_level': self.confidence_level,
            'holding_period': self.holding_period,
            'method': self.method.value,
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'cvar_pct': self.cvar_pct,
            'tail_loss_count': self.tail_loss_count,
        }


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario: StressScenario
    scenario_name: str
    portfolio_loss: float               # 组合损失
    portfolio_loss_pct: float           # 损失百分比
    position_losses: Dict[str, float]   # 各持仓损失
    timestamp: datetime
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario': self.scenario.value,
            'scenario_name': self.scenario_name,
            'portfolio_loss': self.portfolio_loss,
            'portfolio_loss_pct': self.portfolio_loss_pct,
            'position_losses': self.position_losses,
            'timestamp': self.timestamp.isoformat(),
            'assumptions': self.assumptions,
        }


@dataclass
class RiskLimit:
    """风险限额"""
    limit_type: RiskLimitType
    limit_value: float
    current_value: float = 0.0
    utilization: float = 0.0            # 使用率
    warning_threshold: float = 0.8      # 预警阈值
    critical_threshold: float = 0.95    # 临界阈值
    is_breached: bool = False
    is_warning: bool = False

    def update(self, current_value: float):
        """更新当前值"""
        self.current_value = current_value
        if self.limit_value > 0:
            self.utilization = current_value / self.limit_value
        self.is_warning = self.utilization >= self.warning_threshold
        self.is_breached = self.utilization >= 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'limit_type': self.limit_type.value,
            'limit_value': self.limit_value,
            'current_value': self.current_value,
            'utilization': self.utilization,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'is_breached': self.is_breached,
            'is_warning': self.is_warning,
        }


@dataclass
class RiskAlert:
    """风险告警"""
    alert_id: str
    alert_type: RiskLimitType
    severity: AlertSeverity
    message: str
    current_value: float
    limit_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
        }


@dataclass
class RiskMetrics:
    """综合风险指标"""
    timestamp: datetime
    portfolio_value: float
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0             # 年化波动率
    beta: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    leverage: float = 1.0
    concentration: float = 0.0          # 最大持仓集中度
    correlation_avg: float = 0.0        # 平均相关性

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'volatility': self.volatility,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'leverage': self.leverage,
            'concentration': self.concentration,
            'correlation_avg': self.correlation_avg,
        }


@dataclass
class MonitorConfig:
    """监控配置"""
    update_interval_sec: float = 60.0   # 更新间隔
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_holding_period: int = 1         # 持有期(天)
    var_method: VaRMethod = VaRMethod.HISTORICAL
    lookback_days: int = 252            # 历史回溯天数
    monte_carlo_simulations: int = 10000
    enable_stress_testing: bool = True
    enable_alerts: bool = True
    risk_free_rate: float = 0.02        # 无风险利率


# ============== VaR Calculators ==============

class VaRCalculator:
    """VaR计算器基类"""

    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1):
        self.confidence_level = confidence_level
        self.holding_period = holding_period

    def calculate(self, returns: List[float], portfolio_value: float) -> VaRResult:
        """计算VaR"""
        raise NotImplementedError


class HistoricalVaR(VaRCalculator):
    """历史模拟法VaR"""

    def calculate(self, returns: List[float], portfolio_value: float) -> VaRResult:
        if not returns:
            return VaRResult(
                var_value=0.0,
                confidence_level=self.confidence_level,
                holding_period=self.holding_period,
                method=VaRMethod.HISTORICAL,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
            )

        # 排序收益率
        sorted_returns = sorted(returns)

        # 计算分位数位置
        index = int((1 - self.confidence_level) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        # VaR为该分位数对应的损失
        var_return = sorted_returns[index]
        var_value = abs(var_return * portfolio_value)

        # 调整持有期
        if self.holding_period > 1:
            var_value *= math.sqrt(self.holding_period)

        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_pct=var_value / portfolio_value * 100 if portfolio_value > 0 else 0,
        )


class ParametricVaR(VaRCalculator):
    """参数法VaR (方差-协方差法)"""

    # 标准正态分布分位数
    Z_SCORES = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326,
        0.995: 2.576,
    }

    def calculate(self, returns: List[float], portfolio_value: float) -> VaRResult:
        if not returns or len(returns) < 2:
            return VaRResult(
                var_value=0.0,
                confidence_level=self.confidence_level,
                holding_period=self.holding_period,
                method=VaRMethod.PARAMETRIC,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
            )

        # 计算均值和标准差
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        # 获取z分数
        z_score = self.Z_SCORES.get(self.confidence_level, 1.645)

        # 计算VaR
        var_return = mean - z_score * std
        var_value = abs(var_return * portfolio_value)

        # 调整持有期
        if self.holding_period > 1:
            var_value *= math.sqrt(self.holding_period)

        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method=VaRMethod.PARAMETRIC,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_pct=var_value / portfolio_value * 100 if portfolio_value > 0 else 0,
        )


class MonteCarloVaR(VaRCalculator):
    """蒙特卡洛模拟VaR"""

    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1,
                 simulations: int = 10000):
        super().__init__(confidence_level, holding_period)
        self.simulations = simulations

    def calculate(self, returns: List[float], portfolio_value: float) -> VaRResult:
        if not returns or len(returns) < 2:
            return VaRResult(
                var_value=0.0,
                confidence_level=self.confidence_level,
                holding_period=self.holding_period,
                method=VaRMethod.MONTE_CARLO,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
            )

        import random

        # 计算历史均值和标准差
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        # 蒙特卡洛模拟
        simulated_returns = []
        for _ in range(self.simulations):
            # 生成正态分布随机收益率
            sim_return = random.gauss(mean, std)
            # 多日持有期
            for _ in range(self.holding_period - 1):
                sim_return += random.gauss(mean, std)
            simulated_returns.append(sim_return)

        # 排序并取分位数
        sorted_returns = sorted(simulated_returns)
        index = int((1 - self.confidence_level) * len(sorted_returns))
        var_return = sorted_returns[index]
        var_value = abs(var_return * portfolio_value)

        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method=VaRMethod.MONTE_CARLO,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_pct=var_value / portfolio_value * 100 if portfolio_value > 0 else 0,
        )


class CornishFisherVaR(VaRCalculator):
    """Cornish-Fisher扩展VaR (考虑偏度和峰度)"""

    Z_SCORES = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326,
        0.995: 2.576,
    }

    def calculate(self, returns: List[float], portfolio_value: float) -> VaRResult:
        if not returns or len(returns) < 4:
            return VaRResult(
                var_value=0.0,
                confidence_level=self.confidence_level,
                holding_period=self.holding_period,
                method=VaRMethod.CORNISH_FISHER,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
            )

        # 计算矩
        n = len(returns)
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        # 计算偏度和峰度
        m3 = sum((r - mean) ** 3 for r in returns) / n
        m4 = sum((r - mean) ** 4 for r in returns) / n
        skewness = m3 / (std ** 3) if std > 0 else 0
        kurtosis = (m4 / (std ** 4)) - 3 if std > 0 else 0  # 超额峰度

        # 获取z分数
        z = self.Z_SCORES.get(self.confidence_level, 1.645)

        # Cornish-Fisher调整
        z_cf = (z + (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)

        # 计算VaR
        var_return = mean - z_cf * std
        var_value = abs(var_return * portfolio_value)

        # 调整持有期
        if self.holding_period > 1:
            var_value *= math.sqrt(self.holding_period)

        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method=VaRMethod.CORNISH_FISHER,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_pct=var_value / portfolio_value * 100 if portfolio_value > 0 else 0,
        )


# ============== CVaR Calculator ==============

class CVaRCalculator:
    """CVaR (Expected Shortfall) 计算器"""

    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1):
        self.confidence_level = confidence_level
        self.holding_period = holding_period

    def calculate(self, returns: List[float], portfolio_value: float) -> CVaRResult:
        """计算CVaR"""
        if not returns:
            return CVaRResult(
                cvar_value=0.0,
                var_value=0.0,
                confidence_level=self.confidence_level,
                holding_period=self.holding_period,
                method=VaRMethod.HISTORICAL,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
            )

        # 排序收益率
        sorted_returns = sorted(returns)

        # VaR分位数位置
        var_index = int((1 - self.confidence_level) * len(sorted_returns))
        var_index = max(0, min(var_index, len(sorted_returns) - 1))

        # VaR值
        var_return = sorted_returns[var_index]
        var_value = abs(var_return * portfolio_value)

        # CVaR: 超过VaR的平均损失
        tail_returns = sorted_returns[:var_index + 1]
        if tail_returns:
            cvar_return = statistics.mean(tail_returns)
        else:
            cvar_return = var_return

        cvar_value = abs(cvar_return * portfolio_value)

        # 调整持有期
        if self.holding_period > 1:
            var_value *= math.sqrt(self.holding_period)
            cvar_value *= math.sqrt(self.holding_period)

        return CVaRResult(
            cvar_value=cvar_value,
            var_value=var_value,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            cvar_pct=cvar_value / portfolio_value * 100 if portfolio_value > 0 else 0,
            tail_loss_count=len(tail_returns),
        )


# ============== Stress Test Engine ==============

class StressTestEngine:
    """压力测试引擎"""

    # 预设场景参数
    SCENARIO_PARAMS = {
        StressScenario.MARKET_CRASH: {
            'equity_shock': -0.20,          # 股票下跌20%
            'volatility_multiplier': 3.0,   # 波动率放大3倍
            'correlation_increase': 0.3,    # 相关性增加
        },
        StressScenario.INTEREST_RATE_SHOCK: {
            'rate_change_bps': 200,         # 利率上升200基点
            'bond_duration_effect': -0.08,  # 债券价格影响
            'equity_effect': -0.05,
        },
        StressScenario.CURRENCY_CRISIS: {
            'fx_depreciation': -0.15,       # 本币贬值15%
            'equity_effect': -0.10,
            'em_additional_loss': -0.10,    # 新兴市场额外损失
        },
        StressScenario.LIQUIDITY_CRISIS: {
            'liquidity_haircut': -0.10,     # 流动性折价
            'bid_ask_spread_mult': 5.0,     # 买卖价差放大
            'fire_sale_loss': -0.15,
        },
        StressScenario.VOLATILITY_SPIKE: {
            'volatility_multiplier': 4.0,
            'option_gamma_loss': -0.05,
            'correlation_spike': 0.4,
        },
        StressScenario.SECTOR_ROTATION: {
            'growth_sector_loss': -0.15,
            'value_sector_gain': 0.05,
            'sector_dispersion': 0.20,
        },
        StressScenario.CORRELATION_BREAKDOWN: {
            'correlation_change': -0.5,     # 相关性大幅变化
            'diversification_loss': -0.08,
        },
    }

    def __init__(self):
        self._custom_scenarios: Dict[str, Dict[str, float]] = {}

    def add_custom_scenario(self, name: str, params: Dict[str, float]):
        """添加自定义场景"""
        self._custom_scenarios[name] = params

    def run_stress_test(
        self,
        positions: List[PositionRisk],
        portfolio_value: float,
        scenario: StressScenario,
        custom_params: Optional[Dict[str, float]] = None,
    ) -> StressTestResult:
        """运行压力测试"""
        # 获取场景参数
        if scenario == StressScenario.CUSTOM and custom_params:
            params = custom_params
        else:
            params = self.SCENARIO_PARAMS.get(scenario, {})

        # 计算各持仓损失
        position_losses = {}
        total_loss = 0.0

        for pos in positions:
            # 基础冲击
            shock = params.get('equity_shock', 0.0)

            # 波动率调整
            vol_mult = params.get('volatility_multiplier', 1.0)
            vol_adjustment = (vol_mult - 1) * pos.volatility * 0.5

            # 总损失
            loss_pct = shock + vol_adjustment
            loss = pos.market_value * loss_pct

            position_losses[pos.symbol] = loss
            total_loss += loss

        return StressTestResult(
            scenario=scenario,
            scenario_name=scenario.value,
            portfolio_loss=total_loss,
            portfolio_loss_pct=total_loss / portfolio_value * 100 if portfolio_value > 0 else 0,
            position_losses=position_losses,
            timestamp=datetime.now(),
            assumptions=params,
        )

    def run_all_scenarios(
        self,
        positions: List[PositionRisk],
        portfolio_value: float,
    ) -> List[StressTestResult]:
        """运行所有预设场景"""
        results = []
        for scenario in StressScenario:
            if scenario != StressScenario.CUSTOM:
                result = self.run_stress_test(positions, portfolio_value, scenario)
                results.append(result)
        return results


# ============== Portfolio Risk Monitor ==============

class PortfolioRiskMonitor:
    """
    组合风险监控器

    功能:
    - 实时VaR/CVaR计算
    - 压力测试
    - 风险限额监控
    - 风险告警
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()

        # 持仓数据
        self._positions: Dict[str, PositionRisk] = {}
        self._portfolio_value: float = 0.0

        # 收益率历史
        self._returns_history: List[float] = []
        self._value_history: List[Tuple[datetime, float]] = []

        # VaR计算器
        self._var_calculators: Dict[VaRMethod, VaRCalculator] = {
            VaRMethod.HISTORICAL: HistoricalVaR(),
            VaRMethod.PARAMETRIC: ParametricVaR(),
            VaRMethod.MONTE_CARLO: MonteCarloVaR(
                simulations=self.config.monte_carlo_simulations
            ),
            VaRMethod.CORNISH_FISHER: CornishFisherVaR(),
        }

        # CVaR计算器
        self._cvar_calculator = CVaRCalculator()

        # 压力测试引擎
        self._stress_engine = StressTestEngine()

        # 风险限额
        self._limits: Dict[RiskLimitType, RiskLimit] = {}

        # 告警
        self._alerts: List[RiskAlert] = []
        self._alert_callbacks: List[Callable[[RiskAlert], None]] = []
        self._alert_counter = 0

        # 最新风险指标
        self._latest_metrics: Optional[RiskMetrics] = None
        self._latest_var: Dict[float, VaRResult] = {}
        self._latest_cvar: Dict[float, CVaRResult] = {}

        # 监控状态
        self._state = MonitorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 峰值追踪(用于回撤计算)
        self._peak_value: float = 0.0

        logger.info("PortfolioRiskMonitor initialized")

    # ============== Position Management ==============

    def update_position(self, position: PositionRisk):
        """更新持仓"""
        self._positions[position.symbol] = position
        self._update_portfolio_value()

    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self._positions:
            del self._positions[symbol]
            self._update_portfolio_value()

    def clear_positions(self):
        """清空所有持仓"""
        self._positions.clear()
        self._portfolio_value = 0.0

    def get_positions(self) -> List[PositionRisk]:
        """获取所有持仓"""
        return list(self._positions.values())

    def _update_portfolio_value(self):
        """更新组合价值"""
        self._portfolio_value = sum(p.market_value for p in self._positions.values())

        # 更新峰值
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        # 记录价值历史
        self._value_history.append((datetime.now(), self._portfolio_value))

        # 限制历史长度
        max_history = self.config.lookback_days * 24  # 假设每小时更新
        if len(self._value_history) > max_history:
            self._value_history = self._value_history[-max_history:]

    def add_return(self, daily_return: float):
        """添加日收益率"""
        self._returns_history.append(daily_return)

        # 限制历史长度
        if len(self._returns_history) > self.config.lookback_days:
            self._returns_history = self._returns_history[-self.config.lookback_days:]

    # ============== VaR Calculation ==============

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: Optional[VaRMethod] = None,
    ) -> VaRResult:
        """计算VaR"""
        method = method or self.config.var_method

        # 获取计算器
        calculator = self._var_calculators.get(method)
        if calculator is None:
            calculator = self._var_calculators[VaRMethod.HISTORICAL]

        # 设置置信度和持有期
        calculator.confidence_level = confidence_level
        calculator.holding_period = self.config.var_holding_period

        # 计算
        result = calculator.calculate(self._returns_history, self._portfolio_value)

        # 缓存结果
        self._latest_var[confidence_level] = result

        return result

    def calculate_cvar(self, confidence_level: float = 0.95) -> CVaRResult:
        """计算CVaR"""
        self._cvar_calculator.confidence_level = confidence_level
        self._cvar_calculator.holding_period = self.config.var_holding_period

        result = self._cvar_calculator.calculate(
            self._returns_history, self._portfolio_value
        )

        # 缓存结果
        self._latest_cvar[confidence_level] = result

        return result

    def calculate_all_var(self) -> Dict[str, Any]:
        """计算所有置信度的VaR和CVaR"""
        results = {
            'var': {},
            'cvar': {},
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self._portfolio_value,
        }

        for conf in self.config.var_confidence_levels:
            var_result = self.calculate_var(conf)
            cvar_result = self.calculate_cvar(conf)

            results['var'][f'{int(conf*100)}%'] = var_result.to_dict()
            results['cvar'][f'{int(conf*100)}%'] = cvar_result.to_dict()

        return results

    # ============== Stress Testing ==============

    def run_stress_test(
        self,
        scenario: StressScenario,
        custom_params: Optional[Dict[str, float]] = None,
    ) -> StressTestResult:
        """运行压力测试"""
        positions = list(self._positions.values())
        return self._stress_engine.run_stress_test(
            positions, self._portfolio_value, scenario, custom_params
        )

    def run_all_stress_tests(self) -> List[StressTestResult]:
        """运行所有压力测试"""
        positions = list(self._positions.values())
        return self._stress_engine.run_all_scenarios(positions, self._portfolio_value)

    # ============== Risk Limits ==============

    def set_limit(
        self,
        limit_type: RiskLimitType,
        limit_value: float,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        """设置风险限额"""
        self._limits[limit_type] = RiskLimit(
            limit_type=limit_type,
            limit_value=limit_value,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )

    def remove_limit(self, limit_type: RiskLimitType):
        """移除风险限额"""
        if limit_type in self._limits:
            del self._limits[limit_type]

    def get_limits(self) -> Dict[RiskLimitType, RiskLimit]:
        """获取所有风险限额"""
        return dict(self._limits)

    def check_limits(self) -> List[RiskLimit]:
        """检查所有风险限额"""
        breached = []

        for limit_type, limit in self._limits.items():
            current_value = self._get_current_limit_value(limit_type)
            limit.update(current_value)

            if limit.is_breached or limit.is_warning:
                breached.append(limit)

                # 触发告警
                if self.config.enable_alerts:
                    self._trigger_alert(limit)

        return breached

    def _get_current_limit_value(self, limit_type: RiskLimitType) -> float:
        """获取当前限额对应的值"""
        if limit_type == RiskLimitType.VAR_LIMIT:
            var_result = self._latest_var.get(0.95)
            return var_result.var_value if var_result else 0.0

        elif limit_type == RiskLimitType.CVAR_LIMIT:
            cvar_result = self._latest_cvar.get(0.95)
            return cvar_result.cvar_value if cvar_result else 0.0

        elif limit_type == RiskLimitType.DRAWDOWN_LIMIT:
            if self._peak_value > 0:
                return (self._peak_value - self._portfolio_value) / self._peak_value
            return 0.0

        elif limit_type == RiskLimitType.CONCENTRATION_LIMIT:
            if not self._positions or self._portfolio_value == 0:
                return 0.0
            max_weight = max(
                p.market_value / self._portfolio_value
                for p in self._positions.values()
            )
            return max_weight

        elif limit_type == RiskLimitType.EXPOSURE_LIMIT:
            return self._portfolio_value

        elif limit_type == RiskLimitType.LEVERAGE_LIMIT:
            # 简化：假设杠杆=总敞口/净值
            return 1.0  # 需要实际数据

        elif limit_type == RiskLimitType.VOLATILITY_LIMIT:
            if len(self._returns_history) >= 2:
                return statistics.stdev(self._returns_history) * math.sqrt(252)
            return 0.0

        return 0.0

    # ============== Alerts ==============

    def _trigger_alert(self, limit: RiskLimit):
        """触发告警"""
        self._alert_counter += 1

        severity = AlertSeverity.CRITICAL if limit.is_breached else AlertSeverity.WARNING

        alert = RiskAlert(
            alert_id=f"ALERT_{self._alert_counter}",
            alert_type=limit.limit_type,
            severity=severity,
            message=f"{limit.limit_type.value}: {limit.current_value:.2f} / {limit.limit_value:.2f} ({limit.utilization:.1%})",
            current_value=limit.current_value,
            limit_value=limit.limit_value,
            timestamp=datetime.now(),
        )

        self._alerts.append(alert)

        # 触发回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def get_alerts(
        self,
        unacknowledged_only: bool = False,
        severity: Optional[AlertSeverity] = None,
    ) -> List[RiskAlert]:
        """获取告警"""
        alerts = self._alerts

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    # ============== Risk Metrics ==============

    def calculate_metrics(self) -> RiskMetrics:
        """计算综合风险指标"""
        now = datetime.now()

        # 计算VaR/CVaR
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        cvar_99 = self.calculate_cvar(0.99)

        # 计算波动率
        volatility = 0.0
        if len(self._returns_history) >= 2:
            volatility = statistics.stdev(self._returns_history) * math.sqrt(252)

        # 计算回撤
        current_drawdown = 0.0
        if self._peak_value > 0:
            current_drawdown = (self._peak_value - self._portfolio_value) / self._peak_value

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算集中度
        concentration = 0.0
        if self._positions and self._portfolio_value > 0:
            concentration = max(
                p.market_value / self._portfolio_value
                for p in self._positions.values()
            )

        # 计算平均相关性
        correlation_avg = 0.0
        if self._positions:
            correlations = [p.correlation for p in self._positions.values()]
            if correlations:
                correlation_avg = statistics.mean(correlations)

        # 计算Sharpe比率
        sharpe_ratio = 0.0
        if len(self._returns_history) >= 2 and volatility > 0:
            mean_return = statistics.mean(self._returns_history) * 252
            sharpe_ratio = (mean_return - self.config.risk_free_rate) / volatility

        metrics = RiskMetrics(
            timestamp=now,
            portfolio_value=self._portfolio_value,
            var_95=var_95.var_value,
            var_99=var_99.var_value,
            cvar_95=cvar_95.cvar_value,
            cvar_99=cvar_99.cvar_value,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            concentration=concentration,
            correlation_avg=correlation_avg,
        )

        self._latest_metrics = metrics
        return metrics

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self._value_history) < 2:
            return 0.0

        values = [v for _, v in self._value_history]
        peak = values[0]
        max_dd = 0.0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def get_latest_metrics(self) -> Optional[RiskMetrics]:
        """获取最新风险指标"""
        return self._latest_metrics

    # ============== Monitoring ==============

    async def start(self):
        """启动监控"""
        if self._state == MonitorState.RUNNING:
            return

        self._state = MonitorState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Portfolio risk monitor started")

    async def stop(self):
        """停止监控"""
        self._state = MonitorState.IDLE
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Portfolio risk monitor stopped")

    async def _monitor_loop(self):
        """监控循环"""
        while self._state == MonitorState.RUNNING:
            try:
                # 更新风险指标
                self.calculate_metrics()

                # 检查限额
                self.check_limits()

                # 运行压力测试
                if self.config.enable_stress_testing:
                    self.run_all_stress_tests()

                await asyncio.sleep(self.config.update_interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = MonitorState.ERROR
                await asyncio.sleep(5)

    @property
    def state(self) -> MonitorState:
        """获取监控状态"""
        return self._state

    # ============== Statistics ==============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.value,
            'portfolio_value': self._portfolio_value,
            'position_count': len(self._positions),
            'returns_history_length': len(self._returns_history),
            'value_history_length': len(self._value_history),
            'limits_count': len(self._limits),
            'alerts_count': len(self._alerts),
            'unacknowledged_alerts': len([a for a in self._alerts if not a.acknowledged]),
            'latest_metrics': self._latest_metrics.to_dict() if self._latest_metrics else None,
            'config': {
                'update_interval_sec': self.config.update_interval_sec,
                'var_confidence_levels': self.config.var_confidence_levels,
                'var_method': self.config.var_method.value,
                'lookback_days': self.config.lookback_days,
            },
        }


# ============== Factory Functions ==============

def create_risk_monitor(
    update_interval_sec: float = 60.0,
    var_method: str = 'historical',
    lookback_days: int = 252,
    **kwargs
) -> PortfolioRiskMonitor:
    """
    创建组合风险监控器

    Args:
        update_interval_sec: 更新间隔(秒)
        var_method: VaR计算方法
        lookback_days: 历史回溯天数
        **kwargs: 其他配置参数
    """
    try:
        method = VaRMethod(var_method)
    except ValueError:
        method = VaRMethod.HISTORICAL

    config = MonitorConfig(
        update_interval_sec=update_interval_sec,
        var_method=method,
        lookback_days=lookback_days,
        **{k: v for k, v in kwargs.items() if hasattr(MonitorConfig, k)}
    )

    return PortfolioRiskMonitor(config)


def create_position_risk(
    symbol: str,
    quantity: float,
    market_value: float,
    weight: float,
    volatility: float,
    **kwargs
) -> PositionRisk:
    """创建持仓风险数据"""
    return PositionRisk(
        symbol=symbol,
        quantity=quantity,
        market_value=market_value,
        weight=weight,
        volatility=volatility,
        **{k: v for k, v in kwargs.items() if k in [
            'var_contribution', 'cvar_contribution', 'beta',
            'correlation', 'marginal_var', 'component_var'
        ]}
    )


def create_risk_limit(
    limit_type: str,
    limit_value: float,
    warning_threshold: float = 0.8,
    critical_threshold: float = 0.95,
) -> RiskLimit:
    """创建风险限额"""
    try:
        lt = RiskLimitType(limit_type)
    except ValueError:
        lt = RiskLimitType.VAR_LIMIT

    return RiskLimit(
        limit_type=lt,
        limit_value=limit_value,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )


async def quick_risk_analysis(
    positions: List[Dict[str, Any]],
    returns: List[float],
    confidence_levels: List[float] = None,
) -> Dict[str, Any]:
    """
    快速风险分析

    Args:
        positions: 持仓列表
        returns: 历史收益率
        confidence_levels: 置信度列表
    """
    confidence_levels = confidence_levels or [0.95, 0.99]

    monitor = create_risk_monitor()

    # 添加持仓
    total_value = 0.0
    for pos in positions:
        total_value += pos.get('market_value', 0)

    for pos in positions:
        position = create_position_risk(
            symbol=pos.get('symbol', ''),
            quantity=pos.get('quantity', 0),
            market_value=pos.get('market_value', 0),
            weight=pos.get('market_value', 0) / total_value if total_value > 0 else 0,
            volatility=pos.get('volatility', 0.2),
        )
        monitor.update_position(position)

    # 添加收益率
    for r in returns:
        monitor.add_return(r)

    # 计算VaR/CVaR
    results = {
        'portfolio_value': monitor._portfolio_value,
        'var': {},
        'cvar': {},
        'stress_tests': [],
    }

    for conf in confidence_levels:
        var_result = monitor.calculate_var(conf)
        cvar_result = monitor.calculate_cvar(conf)
        results['var'][f'{int(conf*100)}%'] = var_result.to_dict()
        results['cvar'][f'{int(conf*100)}%'] = cvar_result.to_dict()

    # 压力测试
    stress_results = monitor.run_all_stress_tests()
    results['stress_tests'] = [r.to_dict() for r in stress_results]

    # 综合指标
    metrics = monitor.calculate_metrics()
    results['metrics'] = metrics.to_dict()

    return results


__all__ = [
    # Enums
    'VaRMethod',
    'StressScenario',
    'RiskLimitType',
    'AlertSeverity',
    'MonitorState',
    # Data Classes
    'PositionRisk',
    'VaRResult',
    'CVaRResult',
    'StressTestResult',
    'RiskLimit',
    'RiskAlert',
    'RiskMetrics',
    'MonitorConfig',
    # Calculators
    'VaRCalculator',
    'HistoricalVaR',
    'ParametricVaR',
    'MonteCarloVaR',
    'CornishFisherVaR',
    'CVaRCalculator',
    # Stress Test
    'StressTestEngine',
    # Monitor
    'PortfolioRiskMonitor',
    # Factory Functions
    'create_risk_monitor',
    'create_position_risk',
    'create_risk_limit',
    'quick_risk_analysis',
]
