"""
Alpha信号生成器 (Alpha Signal Generator)

实时生成和管理多因子Alpha信号:
- Momentum Alpha (动量Alpha)
- Mean Reversion Alpha (均值回归Alpha)
- Value Alpha (价值Alpha)
- Quality Alpha (质量Alpha)
- Volatility Alpha (波动率Alpha)
- Cross-sectional Alpha (截面Alpha)
- 多因子组合
- 因子权重优化
- 信号标准化
- 信号衰减处理
- IC/IR计算
- 信号覆盖率
- 换手率分析

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

class AlphaType(str, Enum):
    """Alpha类型"""
    MOMENTUM = "momentum"                # 动量Alpha
    MEAN_REVERSION = "mean_reversion"    # 均值回归Alpha
    VALUE = "value"                      # 价值Alpha
    QUALITY = "quality"                  # 质量Alpha
    VOLATILITY = "volatility"            # 波动率Alpha
    CROSS_SECTIONAL = "cross_sectional"  # 截面Alpha


class SignalDirection(str, Enum):
    """信号方向"""
    LONG = "long"      # 做多
    SHORT = "short"    # 做空
    NEUTRAL = "neutral"  # 中性


class SignalStrength(str, Enum):
    """信号强度"""
    VERY_STRONG = "very_strong"      # 非常强 (80-100)
    STRONG = "strong"                # 强 (60-80)
    MODERATE = "moderate"            # 中等 (40-60)
    WEAK = "weak"                    # 弱 (20-40)
    VERY_WEAK = "very_weak"          # 非常弱 (0-20)


class FactorCategory(str, Enum):
    """因子类别"""
    PRICE = "price"              # 价格因子
    VOLUME = "volume"            # 成交量因子
    FUNDAMENTAL = "fundamental"  # 基本面因子
    TECHNICAL = "technical"      # 技术因子
    SENTIMENT = "sentiment"      # 情绪因子
    ALTERNATIVE = "alternative"  # 另类因子


class GeneratorState(str, Enum):
    """生成器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============== Data Classes ==============

@dataclass
class MarketData:
    """
    市场数据

    包含价格、成交量等市场信息
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    turnover: float = 0.0  # 成交额

    # 基本面数据 (可选)
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'turnover': self.turnover,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
        }


@dataclass
class AlphaFactor:
    """
    Alpha因子

    单个因子的计算结果
    """
    name: str                    # 因子名称
    alpha_type: AlphaType        # Alpha类型
    category: FactorCategory     # 因子类别
    value: float                 # 因子值
    zscore: float = 0.0          # 标准化得分
    timestamp: datetime = field(default_factory=datetime.now)

    # 因子质量指标
    ic: float = 0.0              # 信息系数
    ir: float = 0.0              # 信息比率
    coverage: float = 0.0        # 覆盖率

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'alpha_type': self.alpha_type.value,
            'category': self.category.value,
            'value': self.value,
            'zscore': self.zscore,
            'timestamp': self.timestamp.isoformat(),
            'ic': self.ic,
            'ir': self.ir,
            'coverage': self.coverage,
            'metadata': self.metadata,
        }


@dataclass
class FactorWeight:
    """
    因子权重

    记录因子在组合中的权重
    """
    factor_name: str
    weight: float                # 权重 (0-1)
    confidence: float = 1.0      # 置信度
    decay_rate: float = 0.0      # 衰减率
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'factor_name': self.factor_name,
            'weight': self.weight,
            'confidence': self.confidence,
            'decay_rate': self.decay_rate,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class AlphaSignal:
    """
    Alpha信号

    综合多个因子生成的交易信号
    """
    symbol: str
    timestamp: datetime
    signal_direction: SignalDirection
    signal_strength: SignalStrength

    # 信号值
    alpha_value: float           # Alpha值 (-100 to 100)
    confidence: float            # 置信度 (0-1)

    # 贡献因子
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)

    # 预期收益和风险
    expected_return: float = 0.0  # 预期收益率
    expected_volatility: float = 0.0  # 预期波动率
    sharpe_estimate: float = 0.0  # 预期夏普率

    # 衰减信息
    decay_factor: float = 1.0     # 衰减因子 (0-1)
    half_life: float = 5.0        # 半衰期 (天)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_direction': self.signal_direction.value,
            'signal_strength': self.signal_strength.value,
            'alpha_value': self.alpha_value,
            'confidence': self.confidence,
            'contributing_factors': self.contributing_factors,
            'factor_weights': self.factor_weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_estimate': self.sharpe_estimate,
            'decay_factor': self.decay_factor,
            'half_life': self.half_life,
        }


@dataclass
class SignalMetrics:
    """
    信号指标

    评估信号质量的各项指标
    """
    symbol: str
    timestamp: datetime

    # IC相关
    ic: float = 0.0              # 信息系数
    ic_ir: float = 0.0           # IC信息比率
    ic_mean: float = 0.0         # IC均值
    ic_std: float = 0.0          # IC标准差

    # 覆盖率
    coverage_rate: float = 0.0   # 信号覆盖率
    valid_signals: int = 0       # 有效信号数
    total_observations: int = 0  # 总观测数

    # 换手率
    turnover_rate: float = 0.0   # 换手率
    signal_changes: int = 0      # 信号变化次数

    # 因子相关性
    factor_correlation: Dict[str, float] = field(default_factory=dict)

    # 性能
    cumulative_return: float = 0.0  # 累计收益
    annualized_return: float = 0.0  # 年化收益
    volatility: float = 0.0          # 波动率
    sharpe_ratio: float = 0.0        # 夏普率
    max_drawdown: float = 0.0        # 最大回撤

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'ic': self.ic,
            'ic_ir': self.ic_ir,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'coverage_rate': self.coverage_rate,
            'valid_signals': self.valid_signals,
            'total_observations': self.total_observations,
            'turnover_rate': self.turnover_rate,
            'signal_changes': self.signal_changes,
            'factor_correlation': self.factor_correlation,
            'cumulative_return': self.cumulative_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
        }


@dataclass
class AlphaConfig:
    """Alpha配置"""
    # 因子计算周期
    momentum_period: int = 20        # 动量周期
    reversion_period: int = 10       # 均值回归周期
    volatility_period: int = 20      # 波动率周期

    # 标准化参数
    normalize_method: str = "zscore"  # zscore, rank, minmax
    winsorize_quantile: float = 0.025  # 缩尾分位数

    # 权重优化
    weight_method: str = "equal"     # equal, ic_weighted, variance_weighted
    rebalance_frequency: int = 5     # 再平衡频率(天)

    # 信号衰减
    enable_decay: bool = True        # 启用衰减
    decay_half_life: float = 5.0     # 衰减半衰期(天)

    # IC计算
    ic_window: int = 20              # IC计算窗口
    min_ic_samples: int = 10         # 最小IC样本数

    # 覆盖率和换手率
    min_coverage: float = 0.7        # 最小覆盖率
    max_turnover: float = 0.5        # 最大换手率

    # 因子权重
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.25,
        'mean_reversion': 0.20,
        'value': 0.20,
        'quality': 0.15,
        'volatility': 0.10,
        'cross_sectional': 0.10,
    })

    # 监控参数
    update_interval: float = 60.0    # 更新间隔(秒)
    history_size: int = 1000         # 历史数据大小

    def to_dict(self) -> Dict[str, Any]:
        return {
            'momentum_period': self.momentum_period,
            'reversion_period': self.reversion_period,
            'volatility_period': self.volatility_period,
            'normalize_method': self.normalize_method,
            'winsorize_quantile': self.winsorize_quantile,
            'weight_method': self.weight_method,
            'rebalance_frequency': self.rebalance_frequency,
            'enable_decay': self.enable_decay,
            'decay_half_life': self.decay_half_life,
            'ic_window': self.ic_window,
            'min_ic_samples': self.min_ic_samples,
            'min_coverage': self.min_coverage,
            'max_turnover': self.max_turnover,
            'factor_weights': self.factor_weights,
            'update_interval': self.update_interval,
            'history_size': self.history_size,
        }


# ============== Alpha因子计算器 ==============

class AlphaFactorCalculator:
    """
    Alpha因子计算器

    提供各种Alpha因子的计算方法
    """

    @staticmethod
    def calculate_momentum_alpha(prices: List[float], period: int = 20) -> Optional[float]:
        """
        计算动量Alpha

        基于价格动量的Alpha因子

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            动量Alpha值
        """
        if len(prices) < period + 1:
            return None

        # 简单动量: 当前价格 / N期前价格 - 1
        current = prices[-1]
        past = prices[-period - 1]

        if past == 0:
            return None

        momentum = (current / past - 1) * 100
        return momentum

    @staticmethod
    def calculate_mean_reversion_alpha(
        prices: List[float],
        period: int = 10
    ) -> Optional[float]:
        """
        计算均值回归Alpha

        基于价格偏离均值的Alpha因子

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            均值回归Alpha值
        """
        if len(prices) < period:
            return None

        recent_prices = prices[-period:]
        mean_price = statistics.mean(recent_prices)
        current_price = prices[-1]

        if mean_price == 0:
            return None

        # 偏离度 (负值表示低于均值,正值表示高于均值)
        deviation = (current_price - mean_price) / mean_price * 100

        # 均值回归信号: 偏离越大,回归信号越强
        # 负偏离 -> 正信号(买入), 正偏离 -> 负信号(卖出)
        reversion_alpha = -deviation

        return reversion_alpha

    @staticmethod
    def calculate_value_alpha(
        price: float,
        market_cap: float,
        pe_ratio: float,
        pb_ratio: float
    ) -> Optional[float]:
        """
        计算价值Alpha

        基于估值指标的Alpha因子

        Args:
            price: 价格
            market_cap: 市值
            pe_ratio: 市盈率
            pb_ratio: 市净率

        Returns:
            价值Alpha值
        """
        if pe_ratio <= 0 or pb_ratio <= 0:
            return None

        # 估值越低,价值Alpha越高
        # 使用倒数: 1/PE + 1/PB
        value_score = (1 / pe_ratio + 1 / pb_ratio) * 100

        return value_score

    @staticmethod
    def calculate_quality_alpha(
        prices: List[float],
        volumes: List[float],
        period: int = 20
    ) -> Optional[float]:
        """
        计算质量Alpha

        基于价格稳定性和成交量的Alpha因子

        Args:
            prices: 价格序列
            volumes: 成交量序列
            period: 计算周期

        Returns:
            质量Alpha值
        """
        if len(prices) < period or len(volumes) < period:
            return None

        recent_prices = prices[-period:]
        recent_volumes = volumes[-period:]

        # 价格稳定性: 1 / 变异系数
        price_mean = statistics.mean(recent_prices)
        price_std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0

        if price_mean == 0 or price_std == 0:
            price_stability = 1.0
        else:
            cv = price_std / price_mean
            price_stability = 1 / (1 + cv)

        # 成交量稳定性
        volume_mean = statistics.mean(recent_volumes)
        volume_std = statistics.stdev(recent_volumes) if len(recent_volumes) > 1 else 0

        if volume_mean == 0 or volume_std == 0:
            volume_stability = 1.0
        else:
            cv = volume_std / volume_mean
            volume_stability = 1 / (1 + cv)

        # 综合质量得分
        quality_alpha = (price_stability * 0.6 + volume_stability * 0.4) * 100

        return quality_alpha

    @staticmethod
    def calculate_volatility_alpha(
        prices: List[float],
        period: int = 20
    ) -> Optional[float]:
        """
        计算波动率Alpha

        基于价格波动率的Alpha因子

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            波动率Alpha值
        """
        if len(prices) < period + 1:
            return None

        recent_prices = prices[-period:]

        # 计算收益率
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] != 0:
                ret = (recent_prices[i] / recent_prices[i-1] - 1)
                returns.append(ret)

        if len(returns) < 2:
            return None

        # 波动率 (标准差)
        volatility = statistics.stdev(returns)

        # 波动率Alpha: 低波动率 -> 高Alpha
        # 归一化: 假设波动率在0-0.1之间
        volatility_alpha = (0.1 - min(volatility, 0.1)) / 0.1 * 100

        return volatility_alpha

    @staticmethod
    def calculate_cross_sectional_alpha(
        symbol_prices: Dict[str, float],
        target_symbol: str
    ) -> Optional[float]:
        """
        计算截面Alpha

        基于横截面相对表现的Alpha因子

        Args:
            symbol_prices: 所有标的的价格字典
            target_symbol: 目标标的

        Returns:
            截面Alpha值
        """
        if target_symbol not in symbol_prices:
            return None

        prices = list(symbol_prices.values())
        target_price = symbol_prices[target_symbol]

        if not prices:
            return None

        # 计算相对排名
        sorted_prices = sorted(prices)
        rank = sorted_prices.index(target_price)
        percentile = rank / len(sorted_prices)

        # 转换为Alpha: 排名越高,Alpha越高
        cross_sectional_alpha = (percentile - 0.5) * 200  # -100 to 100

        return cross_sectional_alpha


# ============== Alpha信号生成器 ==============

class AlphaSignalGenerator:
    """
    Alpha信号生成器

    功能:
    - 实时计算多种Alpha因子
    - 因子标准化和组合
    - 信号生成和衰减
    - IC/IR计算
    - 信号质量评估
    - 流式数据处理
    """

    def __init__(
        self,
        config: Optional[AlphaConfig] = None,
        signal_callback: Optional[Callable[[AlphaSignal], None]] = None,
        metrics_callback: Optional[Callable[[SignalMetrics], None]] = None,
    ):
        self.config = config or AlphaConfig()
        self.signal_callback = signal_callback
        self.metrics_callback = metrics_callback

        # 状态
        self._state = GeneratorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 市场数据存储
        self._market_data: Dict[str, Deque[MarketData]] = defaultdict(
            lambda: deque(maxlen=self.config.history_size)
        )

        # 因子缓存
        self._factors_cache: Dict[str, Dict[str, AlphaFactor]] = defaultdict(dict)

        # 因子权重
        self._factor_weights: Dict[str, FactorWeight] = {}
        self._initialize_factor_weights()

        # 信号历史
        self._signals_history: Deque[AlphaSignal] = deque(maxlen=self.config.history_size)
        self._current_signals: Dict[str, AlphaSignal] = {}

        # 指标历史
        self._metrics_history: Dict[str, Deque[SignalMetrics]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # IC历史 (用于计算IR)
        self._ic_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.ic_window)
        )

        # 计算器
        self._calculator = AlphaFactorCalculator()

        # 上次再平衡时间
        self._last_rebalance: datetime = datetime.now()

        logger.info("AlphaSignalGenerator initialized")

    @property
    def state(self) -> GeneratorState:
        return self._state

    # ============== 生命周期管理 ==============

    async def start(self) -> None:
        """启动生成器"""
        if self._state == GeneratorState.RUNNING:
            return

        logger.info("Starting Alpha Signal Generator")
        self._state = GeneratorState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Alpha Signal Generator started")

    async def stop(self) -> None:
        """停止生成器"""
        if self._state == GeneratorState.IDLE:
            return

        logger.info("Stopping Alpha Signal Generator")
        self._state = GeneratorState.IDLE

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Alpha Signal Generator stopped")

    async def pause(self) -> None:
        """暂停生成器"""
        if self._state == GeneratorState.RUNNING:
            self._state = GeneratorState.PAUSED
            logger.info("Alpha Signal Generator paused")

    async def resume(self) -> None:
        """恢复生成器"""
        if self._state == GeneratorState.PAUSED:
            self._state = GeneratorState.RUNNING
            logger.info("Alpha Signal Generator resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (GeneratorState.RUNNING, GeneratorState.PAUSED):
            try:
                if self._state == GeneratorState.RUNNING:
                    await self._update_all_signals()
                    await self._check_rebalance()

                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = GeneratorState.ERROR

    # ============== 数据更新 ==============

    async def update_market_data(self, data: MarketData) -> None:
        """
        更新市场数据

        Args:
            data: 市场数据
        """
        symbol = data.symbol
        self._market_data[symbol].append(data)

        # 触发信号更新
        if len(self._market_data[symbol]) >= self.config.momentum_period:
            await self._update_signals_for_symbol(symbol)

    # ============== 因子计算 ==============

    def calculate_momentum_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算动量因子

        Args:
            symbol: 标的符号

        Returns:
            动量因子
        """
        data_list = list(self._market_data[symbol])

        if len(data_list) < self.config.momentum_period + 1:
            return None

        prices = [d.close for d in data_list]

        value = self._calculator.calculate_momentum_alpha(
            prices,
            self.config.momentum_period
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='momentum',
            alpha_type=AlphaType.MOMENTUM,
            category=FactorCategory.TECHNICAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['momentum'] = factor
        return factor

    def calculate_mean_reversion_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算均值回归因子

        Args:
            symbol: 标的符号

        Returns:
            均值回归因子
        """
        data_list = list(self._market_data[symbol])

        if len(data_list) < self.config.reversion_period:
            return None

        prices = [d.close for d in data_list]

        value = self._calculator.calculate_mean_reversion_alpha(
            prices,
            self.config.reversion_period
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='mean_reversion',
            alpha_type=AlphaType.MEAN_REVERSION,
            category=FactorCategory.TECHNICAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['mean_reversion'] = factor
        return factor

    def calculate_value_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算价值因子

        Args:
            symbol: 标的符号

        Returns:
            价值因子
        """
        data_list = list(self._market_data[symbol])

        if not data_list:
            return None

        latest_data = data_list[-1]

        value = self._calculator.calculate_value_alpha(
            latest_data.close,
            latest_data.market_cap,
            latest_data.pe_ratio,
            latest_data.pb_ratio
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='value',
            alpha_type=AlphaType.VALUE,
            category=FactorCategory.FUNDAMENTAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['value'] = factor
        return factor

    def calculate_quality_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算质量因子

        Args:
            symbol: 标的符号

        Returns:
            质量因子
        """
        data_list = list(self._market_data[symbol])

        if len(data_list) < self.config.momentum_period:
            return None

        prices = [d.close for d in data_list]
        volumes = [d.volume for d in data_list]

        value = self._calculator.calculate_quality_alpha(
            prices,
            volumes,
            self.config.momentum_period
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='quality',
            alpha_type=AlphaType.QUALITY,
            category=FactorCategory.TECHNICAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['quality'] = factor
        return factor

    def calculate_volatility_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算波动率因子

        Args:
            symbol: 标的符号

        Returns:
            波动率因子
        """
        data_list = list(self._market_data[symbol])

        if len(data_list) < self.config.volatility_period + 1:
            return None

        prices = [d.close for d in data_list]

        value = self._calculator.calculate_volatility_alpha(
            prices,
            self.config.volatility_period
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='volatility',
            alpha_type=AlphaType.VOLATILITY,
            category=FactorCategory.TECHNICAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['volatility'] = factor
        return factor

    def calculate_cross_sectional_factor(self, symbol: str) -> Optional[AlphaFactor]:
        """
        计算截面因子

        Args:
            symbol: 标的符号

        Returns:
            截面因子
        """
        # 收集所有标的的最新价格
        symbol_prices = {}
        for sym, data_list in self._market_data.items():
            if data_list:
                symbol_prices[sym] = data_list[-1].close

        if symbol not in symbol_prices or len(symbol_prices) < 2:
            return None

        value = self._calculator.calculate_cross_sectional_alpha(
            symbol_prices,
            symbol
        )

        if value is None:
            return None

        factor = AlphaFactor(
            name='cross_sectional',
            alpha_type=AlphaType.CROSS_SECTIONAL,
            category=FactorCategory.TECHNICAL,
            value=value,
            timestamp=datetime.now()
        )

        self._factors_cache[symbol]['cross_sectional'] = factor
        return factor

    # ============== 因子标准化 ==============

    def normalize_factors(self, symbol: str) -> None:
        """
        标准化因子

        对因子进行标准化处理

        Args:
            symbol: 标的符号
        """
        factors = self._factors_cache.get(symbol, {})

        if not factors:
            return

        # 收集所有因子值
        factor_values = {name: factor.value for name, factor in factors.items()}

        # 根据配置选择标准化方法
        if self.config.normalize_method == "zscore":
            self._zscore_normalize(symbol, factor_values)
        elif self.config.normalize_method == "rank":
            self._rank_normalize(symbol, factor_values)
        elif self.config.normalize_method == "minmax":
            self._minmax_normalize(symbol, factor_values)

    def _zscore_normalize(
        self,
        symbol: str,
        factor_values: Dict[str, float]
    ) -> None:
        """Z-score标准化"""
        if not factor_values:
            return

        values = list(factor_values.values())

        if len(values) < 2:
            # 不足2个值,无法计算标准差
            for factor_name in factor_values.keys():
                if factor_name in self._factors_cache[symbol]:
                    self._factors_cache[symbol][factor_name].zscore = 0.0
            return

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            # 标准差为0,所有值相同
            for factor_name in factor_values.keys():
                if factor_name in self._factors_cache[symbol]:
                    self._factors_cache[symbol][factor_name].zscore = 0.0
            return

        # 计算Z-score
        for factor_name, value in factor_values.items():
            zscore = (value - mean) / std

            # Winsorize处理
            if self.config.winsorize_quantile > 0:
                max_z = 3.0  # 3个标准差
                zscore = max(-max_z, min(max_z, zscore))

            if factor_name in self._factors_cache[symbol]:
                self._factors_cache[symbol][factor_name].zscore = zscore

    def _rank_normalize(
        self,
        symbol: str,
        factor_values: Dict[str, float]
    ) -> None:
        """排名标准化"""
        if not factor_values:
            return

        # 排序
        sorted_items = sorted(factor_values.items(), key=lambda x: x[1])
        n = len(sorted_items)

        # 分配排名
        for rank, (factor_name, _) in enumerate(sorted_items):
            # 转换为-1到1的范围
            normalized_rank = (rank / (n - 1) - 0.5) * 2 if n > 1 else 0

            if factor_name in self._factors_cache[symbol]:
                self._factors_cache[symbol][factor_name].zscore = normalized_rank

    def _minmax_normalize(
        self,
        symbol: str,
        factor_values: Dict[str, float]
    ) -> None:
        """Min-Max标准化"""
        if not factor_values:
            return

        values = list(factor_values.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            for factor_name in factor_values.keys():
                if factor_name in self._factors_cache[symbol]:
                    self._factors_cache[symbol][factor_name].zscore = 0.0
            return

        # 归一化到-1到1
        for factor_name, value in factor_values.items():
            normalized = (value - min_val) / (max_val - min_val) * 2 - 1

            if factor_name in self._factors_cache[symbol]:
                self._factors_cache[symbol][factor_name].zscore = normalized

    # ============== 信号生成 ==============

    def generate_alpha_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """
        生成Alpha信号

        综合多个因子生成交易信号

        Args:
            symbol: 标的符号

        Returns:
            Alpha信号
        """
        factors = self._factors_cache.get(symbol, {})

        if not factors:
            return None

        if not self._market_data[symbol]:
            return None

        current_price = self._market_data[symbol][-1].close

        # 标准化因子
        self.normalize_factors(symbol)

        # 计算加权Alpha值
        alpha_value = 0.0
        total_weight = 0.0
        contributing_factors = {}
        factor_weights_used = {}

        for factor_name, factor in factors.items():
            # 获取因子权重
            weight_obj = self._factor_weights.get(factor_name)
            if not weight_obj:
                weight = self.config.factor_weights.get(factor_name, 0.0)
            else:
                weight = weight_obj.weight * weight_obj.confidence

            # 使用标准化后的值
            alpha_value += factor.zscore * weight
            total_weight += weight

            contributing_factors[factor_name] = factor.zscore
            factor_weights_used[factor_name] = weight

        # 标准化Alpha值
        if total_weight > 0:
            alpha_value = (alpha_value / total_weight) * 100  # -100 to 100

        # 确定信号方向
        if alpha_value > 10:
            signal_direction = SignalDirection.LONG
        elif alpha_value < -10:
            signal_direction = SignalDirection.SHORT
        else:
            signal_direction = SignalDirection.NEUTRAL

        # 确定信号强度
        abs_alpha = abs(alpha_value)
        if abs_alpha > 80:
            signal_strength = SignalStrength.VERY_STRONG
        elif abs_alpha > 60:
            signal_strength = SignalStrength.STRONG
        elif abs_alpha > 40:
            signal_strength = SignalStrength.MODERATE
        elif abs_alpha > 20:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK

        # 计算置信度
        confidence = min(1.0, abs_alpha / 100)

        # 计算衰减因子
        decay_factor = 1.0
        if self.config.enable_decay:
            # 检查上次信号时间
            prev_signal = self._current_signals.get(symbol)
            if prev_signal:
                time_elapsed = (datetime.now() - prev_signal.timestamp).total_seconds() / 86400  # 天
                decay_factor = math.exp(-math.log(2) * time_elapsed / self.config.decay_half_life)

        # 估计预期收益和风险
        expected_return = alpha_value / 100 * 0.01  # 简化估计
        expected_volatility = 0.02  # 简化估计
        sharpe_estimate = expected_return / expected_volatility if expected_volatility > 0 else 0

        signal = AlphaSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            alpha_value=alpha_value,
            confidence=confidence,
            contributing_factors=contributing_factors,
            factor_weights=factor_weights_used,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_estimate=sharpe_estimate,
            decay_factor=decay_factor,
            half_life=self.config.decay_half_life
        )

        # 更新当前信号
        self._current_signals[symbol] = signal
        self._signals_history.append(signal)

        # 触发回调
        if self.signal_callback:
            self.signal_callback(signal)

        return signal

    # ============== IC/IR计算 ==============

    def calculate_ic(self, symbol: str) -> Optional[float]:
        """
        计算信息系数 (IC)

        衡量因子预测能力

        Args:
            symbol: 标的符号

        Returns:
            IC值
        """
        signals = [s for s in self._signals_history if s.symbol == symbol]

        if len(signals) < self.config.min_ic_samples:
            return None

        # 取最近的窗口
        recent_signals = signals[-self.config.ic_window:]

        # 计算预测值和实际收益的相关性
        predictions = [s.alpha_value for s in recent_signals]

        # 简化: 使用下一期的收益作为实际值
        # 实际应用中需要真实的未来收益数据
        data_list = list(self._market_data[symbol])
        if len(data_list) < len(recent_signals) + 1:
            return None

        actual_returns = []
        for i in range(len(recent_signals)):
            if i + 1 < len(data_list):
                ret = (data_list[i + 1].close / data_list[i].close - 1) * 100
                actual_returns.append(ret)

        if len(predictions) != len(actual_returns) or len(predictions) < 2:
            return None

        # 计算相关系数
        ic = self._calculate_correlation(predictions, actual_returns)

        # 记录IC历史
        if ic is not None:
            self._ic_history[symbol].append(ic)

        return ic

    def calculate_ir(self, symbol: str) -> Optional[float]:
        """
        计算信息比率 (IR)

        IR = IC均值 / IC标准差

        Args:
            symbol: 标的符号

        Returns:
            IR值
        """
        ic_list = list(self._ic_history.get(symbol, []))

        if len(ic_list) < self.config.min_ic_samples:
            return None

        ic_mean = statistics.mean(ic_list)
        ic_std = statistics.stdev(ic_list) if len(ic_list) > 1 else 0

        if ic_std == 0:
            return None

        ir = ic_mean / ic_std
        return ir

    def _calculate_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> Optional[float]:
        """计算相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return None

        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

        std_x = statistics.stdev(x)
        std_y = statistics.stdev(y)

        if std_x == 0 or std_y == 0:
            return None

        correlation = covariance / (std_x * std_y)
        return max(-1.0, min(1.0, correlation))

    # ============== 信号评估 ==============

    def calculate_signal_metrics(self, symbol: str) -> Optional[SignalMetrics]:
        """
        计算信号指标

        Args:
            symbol: 标的符号

        Returns:
            信号指标
        """
        signals = [s for s in self._signals_history if s.symbol == symbol]

        if not signals:
            return None

        # IC相关
        ic = self.calculate_ic(symbol)
        ir = self.calculate_ir(symbol)

        ic_list = list(self._ic_history.get(symbol, []))
        ic_mean = statistics.mean(ic_list) if ic_list else 0.0
        ic_std = statistics.stdev(ic_list) if len(ic_list) > 1 else 0.0

        # 覆盖率
        valid_signals = len([s for s in signals if s.signal_direction != SignalDirection.NEUTRAL])
        total_observations = len(signals)
        coverage_rate = valid_signals / total_observations if total_observations > 0 else 0.0

        # 换手率
        signal_changes = 0
        for i in range(1, len(signals)):
            if signals[i].signal_direction != signals[i-1].signal_direction:
                signal_changes += 1
        turnover_rate = signal_changes / len(signals) if len(signals) > 0 else 0.0

        # 因子相关性 (简化)
        factor_correlation = {}
        factors = self._factors_cache.get(symbol, {})
        if len(factors) > 1:
            factor_names = list(factors.keys())
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    name1, name2 = factor_names[i], factor_names[j]
                    # 简化: 使用固定值
                    factor_correlation[f"{name1}_{name2}"] = 0.0

        # 性能指标 (简化)
        cumulative_return = sum(s.expected_return for s in signals)

        metrics = SignalMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            ic=ic or 0.0,
            ic_ir=ir or 0.0,
            ic_mean=ic_mean,
            ic_std=ic_std,
            coverage_rate=coverage_rate,
            valid_signals=valid_signals,
            total_observations=total_observations,
            turnover_rate=turnover_rate,
            signal_changes=signal_changes,
            factor_correlation=factor_correlation,
            cumulative_return=cumulative_return,
        )

        # 记录指标历史
        self._metrics_history[symbol].append(metrics)

        # 触发回调
        if self.metrics_callback:
            self.metrics_callback(metrics)

        return metrics

    # ============== 因子权重管理 ==============

    def _initialize_factor_weights(self) -> None:
        """初始化因子权重"""
        for factor_name, weight in self.config.factor_weights.items():
            self._factor_weights[factor_name] = FactorWeight(
                factor_name=factor_name,
                weight=weight,
                confidence=1.0
            )

    async def _check_rebalance(self) -> None:
        """检查是否需要再平衡"""
        time_since_rebalance = (datetime.now() - self._last_rebalance).total_seconds() / 86400

        if time_since_rebalance >= self.config.rebalance_frequency:
            await self._rebalance_weights()
            self._last_rebalance = datetime.now()

    async def _rebalance_weights(self) -> None:
        """再平衡因子权重"""
        if self.config.weight_method == "equal":
            # 等权重
            n_factors = len(self.config.factor_weights)
            if n_factors > 0:
                equal_weight = 1.0 / n_factors
                for factor_name in self.config.factor_weights.keys():
                    if factor_name in self._factor_weights:
                        self._factor_weights[factor_name].weight = equal_weight

        elif self.config.weight_method == "ic_weighted":
            # IC加权
            await self._rebalance_by_ic()

        elif self.config.weight_method == "variance_weighted":
            # 方差加权
            await self._rebalance_by_variance()

    async def _rebalance_by_ic(self) -> None:
        """基于IC再平衡"""
        # 收集所有标的的IC
        factor_ics = defaultdict(list)

        for symbol in self._market_data.keys():
            factors = self._factors_cache.get(symbol, {})
            for factor_name in factors.keys():
                ic = self.calculate_ic(symbol)
                if ic is not None:
                    factor_ics[factor_name].append(abs(ic))

        # 计算平均IC
        avg_ics = {}
        for factor_name, ics in factor_ics.items():
            if ics:
                avg_ics[factor_name] = statistics.mean(ics)

        # 归一化权重
        total_ic = sum(avg_ics.values())
        if total_ic > 0:
            for factor_name, ic in avg_ics.items():
                if factor_name in self._factor_weights:
                    self._factor_weights[factor_name].weight = ic / total_ic

    async def _rebalance_by_variance(self) -> None:
        """基于方差再平衡 (低方差高权重)"""
        # 简化实现
        pass

    # ============== 更新逻辑 ==============

    async def _update_all_signals(self) -> None:
        """更新所有标的的信号"""
        for symbol in list(self._market_data.keys()):
            await self._update_signals_for_symbol(symbol)

    async def _update_signals_for_symbol(self, symbol: str) -> None:
        """更新特定标的的信号"""
        # 计算所有因子
        self.calculate_momentum_factor(symbol)
        self.calculate_mean_reversion_factor(symbol)
        self.calculate_value_factor(symbol)
        self.calculate_quality_factor(symbol)
        self.calculate_volatility_factor(symbol)
        self.calculate_cross_sectional_factor(symbol)

        # 生成信号
        self.generate_alpha_signal(symbol)

        # 计算指标
        self.calculate_signal_metrics(symbol)

    # ============== 查询方法 ==============

    def get_current_signal(self, symbol: str) -> Optional[AlphaSignal]:
        """获取当前信号"""
        return self._current_signals.get(symbol)

    def get_signal_history(self, symbol: str, limit: int = 100) -> List[AlphaSignal]:
        """获取信号历史"""
        signals = [s for s in self._signals_history if s.symbol == symbol]
        return signals[-limit:]

    def get_factor(self, symbol: str, factor_name: str) -> Optional[AlphaFactor]:
        """获取因子"""
        return self._factors_cache.get(symbol, {}).get(factor_name)

    def get_all_factors(self, symbol: str) -> Dict[str, AlphaFactor]:
        """获取所有因子"""
        return self._factors_cache.get(symbol, {})

    def get_factor_weights(self) -> Dict[str, FactorWeight]:
        """获取因子权重"""
        return self._factor_weights.copy()

    def get_metrics(self, symbol: str) -> Optional[SignalMetrics]:
        """获取最新指标"""
        metrics_list = self._metrics_history.get(symbol, [])
        return metrics_list[-1] if metrics_list else None

    def get_metrics_history(self, symbol: str, limit: int = 50) -> List[SignalMetrics]:
        """获取指标历史"""
        metrics_list = list(self._metrics_history.get(symbol, []))
        return metrics_list[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_signals = len(self._signals_history)
        active_symbols = len(self._market_data)

        # 平均覆盖率
        coverage_rates = []
        for symbol in self._market_data.keys():
            metrics = self.get_metrics(symbol)
            if metrics:
                coverage_rates.append(metrics.coverage_rate)
        avg_coverage = statistics.mean(coverage_rates) if coverage_rates else 0.0

        # 平均换手率
        turnover_rates = []
        for symbol in self._market_data.keys():
            metrics = self.get_metrics(symbol)
            if metrics:
                turnover_rates.append(metrics.turnover_rate)
        avg_turnover = statistics.mean(turnover_rates) if turnover_rates else 0.0

        return {
            'state': self._state.value,
            'active_symbols': active_symbols,
            'total_signals': total_signals,
            'total_factors': sum(len(f) for f in self._factors_cache.values()),
            'avg_coverage_rate': avg_coverage,
            'avg_turnover_rate': avg_turnover,
            'factor_weights': {name: w.weight for name, w in self._factor_weights.items()},
            'last_rebalance': self._last_rebalance.isoformat(),
            'config': self.config.to_dict(),
        }


# ============== 工厂函数 ==============

def create_alpha_generator(
    signal_callback: Optional[Callable[[AlphaSignal], None]] = None,
    **config_kwargs
) -> AlphaSignalGenerator:
    """
    创建Alpha信号生成器

    Args:
        signal_callback: 信号回调函数
        **config_kwargs: 配置参数

    Returns:
        Alpha信号生成器实例
    """
    config = AlphaConfig(**config_kwargs)
    return AlphaSignalGenerator(
        config=config,
        signal_callback=signal_callback
    )


def create_alpha_factor(
    name: str,
    alpha_type: AlphaType,
    category: FactorCategory,
    value: float,
    **kwargs
) -> AlphaFactor:
    """
    创建Alpha因子

    Args:
        name: 因子名称
        alpha_type: Alpha类型
        category: 因子类别
        value: 因子值
        **kwargs: 其他参数

    Returns:
        Alpha因子实例
    """
    return AlphaFactor(
        name=name,
        alpha_type=alpha_type,
        category=category,
        value=value,
        **kwargs
    )


def quick_alpha_signal(
    symbol: str,
    alpha_value: float,
    confidence: float = 0.8
) -> AlphaSignal:
    """
    快速创建Alpha信号

    Args:
        symbol: 标的符号
        alpha_value: Alpha值
        confidence: 置信度

    Returns:
        Alpha信号实例
    """
    # 确定方向
    if alpha_value > 0:
        direction = SignalDirection.LONG
    elif alpha_value < 0:
        direction = SignalDirection.SHORT
    else:
        direction = SignalDirection.NEUTRAL

    # 确定强度
    abs_value = abs(alpha_value)
    if abs_value > 80:
        strength = SignalStrength.VERY_STRONG
    elif abs_value > 60:
        strength = SignalStrength.STRONG
    elif abs_value > 40:
        strength = SignalStrength.MODERATE
    elif abs_value > 20:
        strength = SignalStrength.WEAK
    else:
        strength = SignalStrength.VERY_WEAK

    return AlphaSignal(
        symbol=symbol,
        timestamp=datetime.now(),
        signal_direction=direction,
        signal_strength=strength,
        alpha_value=alpha_value,
        confidence=confidence
    )


__all__ = [
    # Enums
    'AlphaType',
    'SignalDirection',
    'SignalStrength',
    'FactorCategory',
    'GeneratorState',
    # Data Classes
    'MarketData',
    'AlphaFactor',
    'FactorWeight',
    'AlphaSignal',
    'SignalMetrics',
    'AlphaConfig',
    # Main Classes
    'AlphaFactorCalculator',
    'AlphaSignalGenerator',
    # Factory Functions
    'create_alpha_generator',
    'create_alpha_factor',
    'quick_alpha_signal',
]
