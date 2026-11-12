"""
执行质量监控器 (Execution Quality Monitor)

机构级订单执行质量分析和监控系统，提供:
- Implementation Shortfall 实施缺口分析
- VWAP Slippage VWAP滑点计算
- Arrival Price Benchmark 到达价格基准
- Market Impact Cost 市场冲击成本
- Timing Cost 时机成本分解
- Opportunity Cost 机会成本评估
- Real-time execution tracking 实时执行追踪
- Broker/Algo comparison 经纪商/算法比较
- Execution alerts 执行异常告警

作者: TraderX 量化团队
Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging
import math
import statistics
import uuid

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class BenchmarkType(str, Enum):
    """基准类型"""
    ARRIVAL_PRICE = "arrival_price"         # 到达价格
    VWAP = "vwap"                           # 成交量加权平均价
    TWAP = "twap"                           # 时间加权平均价
    CLOSING_PRICE = "closing_price"         # 收盘价
    OPEN_PRICE = "open_price"               # 开盘价
    MIDPOINT = "midpoint"                   # 买卖中间价


class ExecutionSide(str, Enum):
    """执行方向"""
    BUY = "buy"
    SELL = "sell"


class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"                     # 待执行
    PARTIAL = "partial"                     # 部分成交
    FILLED = "filled"                       # 完全成交
    CANCELLED = "cancelled"                 # 已取消
    REJECTED = "rejected"                   # 已拒绝


class AlertType(str, Enum):
    """告警类型"""
    HIGH_SLIPPAGE = "high_slippage"         # 高滑点
    HIGH_IMPACT = "high_impact"             # 高冲击成本
    SLOW_EXECUTION = "slow_execution"       # 执行缓慢
    ADVERSE_SELECTION = "adverse_selection" # 逆向选择
    PRICE_DEVIATION = "price_deviation"     # 价格偏离
    VOLUME_ANOMALY = "volume_anomaly"       # 成交量异常


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitorState(str, Enum):
    """监控器状态"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据类 ====================

@dataclass
class ExecutionRecord:
    """执行记录"""
    execution_id: str
    symbol: str
    side: ExecutionSide

    # 订单信息
    order_id: str
    order_quantity: float                   # 订单总量
    order_time: datetime                    # 下单时间

    # 执行信息
    executed_quantity: float = 0.0          # 已执行数量
    executed_price: float = 0.0             # 平均执行价格
    execution_time: Optional[datetime] = None  # 执行完成时间
    status: ExecutionStatus = ExecutionStatus.PENDING

    # 基准价格
    arrival_price: float = 0.0              # 到达价格(决策时刻价格)
    decision_time: Optional[datetime] = None  # 决策时间

    # 市场数据
    vwap_price: float = 0.0                 # VWAP价格
    closing_price: float = 0.0              # 收盘价
    market_volume: float = 0.0              # 市场总成交量

    # 经纪商/算法信息
    broker: str = ""                        # 经纪商
    algo_name: str = ""                     # 算法名称

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_id': self.order_id,
            'order_quantity': self.order_quantity,
            'order_time': self.order_time.isoformat(),
            'executed_quantity': self.executed_quantity,
            'executed_price': self.executed_price,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'status': self.status.value,
            'arrival_price': self.arrival_price,
            'decision_time': self.decision_time.isoformat() if self.decision_time else None,
            'vwap_price': self.vwap_price,
            'closing_price': self.closing_price,
            'market_volume': self.market_volume,
            'broker': self.broker,
            'algo_name': self.algo_name,
            'metadata': self.metadata,
        }


@dataclass
class CostBreakdown:
    """成本分解"""
    execution_id: str
    symbol: str

    # 总成本
    total_cost: float                       # 总交易成本(bps)
    total_cost_amount: float                # 总成本金额

    # 成本组成部分
    market_impact: float = 0.0              # 市场冲击成本(bps)
    timing_cost: float = 0.0                # 时机成本(bps)
    opportunity_cost: float = 0.0           # 机会成本(bps)
    spread_cost: float = 0.0                # 价差成本(bps)
    commission: float = 0.0                 # 佣金成本(bps)

    # 滑点
    slippage_bps: float = 0.0               # 滑点(bps)
    slippage_amount: float = 0.0            # 滑点金额

    # 时间戳
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'symbol': self.symbol,
            'total_cost': self.total_cost,
            'total_cost_amount': self.total_cost_amount,
            'market_impact': self.market_impact,
            'timing_cost': self.timing_cost,
            'opportunity_cost': self.opportunity_cost,
            'spread_cost': self.spread_cost,
            'commission': self.commission,
            'slippage_bps': self.slippage_bps,
            'slippage_amount': self.slippage_amount,
            'calculated_at': self.calculated_at.isoformat(),
        }


@dataclass
class QualityMetrics:
    """质量指标"""
    execution_id: str
    symbol: str
    benchmark_type: BenchmarkType

    # 实施缺口 (Implementation Shortfall)
    implementation_shortfall: float = 0.0   # 实施缺口(bps)
    is_components: Dict[str, float] = field(default_factory=dict)  # IS组成部分

    # VWAP相关
    vwap_slippage: float = 0.0              # VWAP滑点(bps)
    vwap_participation_rate: float = 0.0    # VWAP参与率

    # 到达价格基准
    arrival_price_slippage: float = 0.0     # 到达价格滑点(bps)

    # 执行效率
    fill_rate: float = 0.0                  # 成交率(0-1)
    execution_duration: float = 0.0         # 执行时长(秒)
    average_fill_size: float = 0.0          # 平均成交规模
    num_fills: int = 0                      # 成交次数

    # 市场影响
    price_impact: float = 0.0               # 价格影响(bps)
    volume_participation: float = 0.0       # 成交量参与率

    # 综合得分
    quality_score: float = 0.0              # 质量得分(0-100)

    # 时间戳
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'symbol': self.symbol,
            'benchmark_type': self.benchmark_type.value,
            'implementation_shortfall': self.implementation_shortfall,
            'is_components': self.is_components,
            'vwap_slippage': self.vwap_slippage,
            'vwap_participation_rate': self.vwap_participation_rate,
            'arrival_price_slippage': self.arrival_price_slippage,
            'fill_rate': self.fill_rate,
            'execution_duration': self.execution_duration,
            'average_fill_size': self.average_fill_size,
            'num_fills': self.num_fills,
            'price_impact': self.price_impact,
            'volume_participation': self.volume_participation,
            'quality_score': self.quality_score,
            'calculated_at': self.calculated_at.isoformat(),
        }


@dataclass
class ExecutionAlert:
    """执行告警"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity

    execution_id: str
    symbol: str

    # 告警信息
    message: str
    trigger_value: float                    # 触发值
    threshold_value: float                  # 阈值

    # 建议
    recommendation: str = ""

    # 时间戳
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'execution_id': self.execution_id,
            'symbol': self.symbol,
            'message': self.message,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'recommendation': self.recommendation,
            'triggered_at': self.triggered_at.isoformat(),
            'acknowledged': self.acknowledged,
        }


@dataclass
class ExecutionStatistics:
    """执行统计"""
    period_start: datetime
    period_end: datetime

    # 基本统计
    total_executions: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0

    # 成本统计
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0
    avg_impact_bps: float = 0.0

    # 效率统计
    avg_fill_rate: float = 0.0
    avg_execution_time: float = 0.0         # 平均执行时长(秒)

    # 质量统计
    avg_quality_score: float = 0.0
    executions_above_threshold: int = 0     # 超过阈值的执行数

    # 按经纪商统计
    broker_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 按算法统计
    algo_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 按时段统计
    hourly_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_executions': self.total_executions,
            'total_volume': self.total_volume,
            'total_value': self.total_value,
            'avg_slippage_bps': self.avg_slippage_bps,
            'median_slippage_bps': self.median_slippage_bps,
            'std_slippage_bps': self.std_slippage_bps,
            'avg_impact_bps': self.avg_impact_bps,
            'avg_fill_rate': self.avg_fill_rate,
            'avg_execution_time': self.avg_execution_time,
            'avg_quality_score': self.avg_quality_score,
            'executions_above_threshold': self.executions_above_threshold,
            'broker_stats': self.broker_stats,
            'algo_stats': self.algo_stats,
            'hourly_stats': self.hourly_stats,
        }


@dataclass
class MonitorConfig:
    """监控器配置"""
    # 告警阈值
    high_slippage_threshold: float = 10.0       # 高滑点阈值(bps)
    high_impact_threshold: float = 15.0         # 高冲击阈值(bps)
    slow_execution_threshold: float = 300.0     # 慢执行阈值(秒)
    adverse_selection_threshold: float = 20.0   # 逆向选择阈值(bps)

    # 默认基准
    default_benchmark: BenchmarkType = BenchmarkType.ARRIVAL_PRICE

    # 统计周期
    statistics_window: int = 100                # 统计窗口大小
    history_retention_days: int = 30            # 历史保留天数

    # 实时监控
    monitor_interval: float = 1.0               # 监控间隔(秒)
    enable_alerts: bool = True                  # 启用告警

    # 成本计算参数
    commission_rate: float = 0.001              # 佣金费率
    spread_estimate: float = 0.0001             # 价差估计

    def to_dict(self) -> Dict[str, Any]:
        return {
            'high_slippage_threshold': self.high_slippage_threshold,
            'high_impact_threshold': self.high_impact_threshold,
            'slow_execution_threshold': self.slow_execution_threshold,
            'adverse_selection_threshold': self.adverse_selection_threshold,
            'default_benchmark': self.default_benchmark.value,
            'statistics_window': self.statistics_window,
            'history_retention_days': self.history_retention_days,
            'monitor_interval': self.monitor_interval,
            'enable_alerts': self.enable_alerts,
            'commission_rate': self.commission_rate,
            'spread_estimate': self.spread_estimate,
        }


# ==================== 核心计算器 ====================

class QualityCalculator:
    """质量计算器"""

    @staticmethod
    def calculate_implementation_shortfall(
        record: ExecutionRecord,
        benchmark_price: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算实施缺口 (Implementation Shortfall)
        IS = (实际执行价格 - 基准价格) / 基准价格 * 方向
        """
        if benchmark_price <= 0 or record.executed_quantity <= 0:
            return 0.0, {}

        # 计算价格差异(考虑方向)
        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (record.executed_price - benchmark_price) * direction

        # 转换为bps
        is_bps = (price_diff / benchmark_price) * 10000

        # 组成部分分解
        components = {
            'execution_cost': is_bps,  # 简化版本，实际应该更细致分解
        }

        return is_bps, components

    @staticmethod
    def calculate_vwap_slippage(
        record: ExecutionRecord,
    ) -> float:
        """计算VWAP滑点"""
        if record.vwap_price <= 0 or record.executed_quantity <= 0:
            return 0.0

        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (record.executed_price - record.vwap_price) * direction

        slippage_bps = (price_diff / record.vwap_price) * 10000

        return slippage_bps

    @staticmethod
    def calculate_arrival_price_slippage(
        record: ExecutionRecord,
    ) -> float:
        """计算到达价格滑点"""
        if record.arrival_price <= 0 or record.executed_quantity <= 0:
            return 0.0

        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (record.executed_price - record.arrival_price) * direction

        slippage_bps = (price_diff / record.arrival_price) * 10000

        return slippage_bps

    @staticmethod
    def calculate_market_impact(
        record: ExecutionRecord,
        initial_price: float,
        final_price: float,
    ) -> float:
        """
        计算市场冲击成本
        Impact = (执行后价格 - 执行前价格) / 执行前价格
        """
        if initial_price <= 0:
            return 0.0

        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_change = (final_price - initial_price) * direction

        impact_bps = (price_change / initial_price) * 10000

        return max(0.0, impact_bps)  # 冲击成本应为正值

    @staticmethod
    def calculate_timing_cost(
        record: ExecutionRecord,
        decision_price: float,
        order_price: float,
    ) -> float:
        """
        计算时机成本
        从决策到下单之间的价格变化
        """
        if decision_price <= 0:
            return 0.0

        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (order_price - decision_price) * direction

        timing_bps = (price_diff / decision_price) * 10000

        return timing_bps

    @staticmethod
    def calculate_opportunity_cost(
        record: ExecutionRecord,
        target_quantity: float,
        closing_price: float,
    ) -> float:
        """
        计算机会成本
        未成交部分的成本
        """
        unfilled_quantity = target_quantity - record.executed_quantity

        if unfilled_quantity <= 0 or record.arrival_price <= 0:
            return 0.0

        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (closing_price - record.arrival_price) * direction

        opp_cost_bps = (price_diff / record.arrival_price) * 10000

        # 加权未成交比例
        fill_rate = record.executed_quantity / target_quantity if target_quantity > 0 else 1.0
        weighted_cost = opp_cost_bps * (1 - fill_rate)

        return weighted_cost

    @staticmethod
    def calculate_quality_score(
        slippage_bps: float,
        impact_bps: float,
        fill_rate: float,
        execution_duration: float,
        target_duration: float = 300.0,
    ) -> float:
        """
        计算综合质量得分 (0-100)
        得分越高，执行质量越好
        """
        # 滑点得分 (0-40分)
        slippage_score = max(0, 40 - abs(slippage_bps) * 2)

        # 冲击得分 (0-30分)
        impact_score = max(0, 30 - impact_bps)

        # 成交率得分 (0-20分)
        fill_score = fill_rate * 20

        # 执行时间得分 (0-10分)
        time_ratio = min(execution_duration / target_duration, 2.0)
        time_score = max(0, 10 - abs(time_ratio - 1.0) * 10)

        total_score = slippage_score + impact_score + fill_score + time_score

        return max(0.0, min(100.0, total_score))


# ==================== 核心监控器 ====================

class ExecutionQualityMonitor:
    """
    执行质量监控器

    功能:
    - 实时执行追踪
    - 质量指标计算
    - 成本分解分析
    - 执行告警
    - 统计分析
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()

        # 计算器
        self._calculator = QualityCalculator()

        # 数据存储
        self._executions: Dict[str, ExecutionRecord] = {}
        self._quality_metrics: Dict[str, QualityMetrics] = {}
        self._cost_breakdowns: Dict[str, CostBreakdown] = {}
        self._alerts: deque = deque(maxlen=1000)

        # 按symbol索引
        self._executions_by_symbol: Dict[str, List[str]] = defaultdict(list)

        # 状态
        self._state = MonitorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 回调
        self._on_alert: Optional[Callable[[ExecutionAlert], None]] = None
        self._on_execution_complete: Optional[Callable[[ExecutionRecord, QualityMetrics], None]] = None

        # 统计
        self.stats = {
            'total_executions': 0,
            'alerts_triggered': 0,
            'avg_quality_score': 0.0,
        }

        logger.info("ExecutionQualityMonitor initialized")

    # ==================== 执行记录管理 ====================

    def add_execution(self, record: ExecutionRecord) -> str:
        """添加执行记录"""
        execution_id = record.execution_id

        self._executions[execution_id] = record
        self._executions_by_symbol[record.symbol].append(execution_id)

        self.stats['total_executions'] += 1

        logger.info(f"Added execution: {execution_id} for {record.symbol}")

        return execution_id

    def update_execution(
        self,
        execution_id: str,
        executed_quantity: Optional[float] = None,
        executed_price: Optional[float] = None,
        status: Optional[ExecutionStatus] = None,
        **kwargs
    ):
        """更新执行记录"""
        if execution_id not in self._executions:
            raise ValueError(f"Execution {execution_id} not found")

        record = self._executions[execution_id]

        if executed_quantity is not None:
            record.executed_quantity = executed_quantity

        if executed_price is not None:
            record.executed_price = executed_price

        if status is not None:
            record.status = status
            if status == ExecutionStatus.FILLED:
                record.execution_time = datetime.now()

        # 更新其他字段
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """获取执行记录"""
        return self._executions.get(execution_id)

    def get_executions_by_symbol(self, symbol: str) -> List[ExecutionRecord]:
        """获取指定品种的执行记录"""
        execution_ids = self._executions_by_symbol.get(symbol, [])
        return [self._executions[eid] for eid in execution_ids if eid in self._executions]

    # ==================== 质量计算 ====================

    def calculate_quality_metrics(
        self,
        execution_id: str,
        benchmark_type: Optional[BenchmarkType] = None,
    ) -> QualityMetrics:
        """计算质量指标"""
        record = self.get_execution(execution_id)
        if not record:
            raise ValueError(f"Execution {execution_id} not found")

        benchmark_type = benchmark_type or self.config.default_benchmark

        # 选择基准价格
        if benchmark_type == BenchmarkType.ARRIVAL_PRICE:
            benchmark_price = record.arrival_price
        elif benchmark_type == BenchmarkType.VWAP:
            benchmark_price = record.vwap_price
        else:
            benchmark_price = record.arrival_price

        # 计算实施缺口
        is_value, is_components = self._calculator.calculate_implementation_shortfall(
            record, benchmark_price
        )

        # 计算VWAP滑点
        vwap_slippage = self._calculator.calculate_vwap_slippage(record)

        # 计算到达价格滑点
        arrival_slippage = self._calculator.calculate_arrival_price_slippage(record)

        # 计算执行效率指标
        fill_rate = record.executed_quantity / record.order_quantity if record.order_quantity > 0 else 0.0

        execution_duration = 0.0
        if record.execution_time and record.order_time:
            execution_duration = (record.execution_time - record.order_time).total_seconds()

        # 市场参与率
        volume_participation = 0.0
        if record.market_volume > 0:
            volume_participation = record.executed_quantity / record.market_volume

        # 计算价格影响
        price_impact = 0.0
        if record.arrival_price > 0 and record.executed_price > 0:
            direction = 1 if record.side == ExecutionSide.BUY else -1
            price_change = (record.executed_price - record.arrival_price) * direction
            price_impact = (price_change / record.arrival_price) * 10000

        # 计算质量得分
        quality_score = self._calculator.calculate_quality_score(
            slippage_bps=arrival_slippage,
            impact_bps=price_impact,
            fill_rate=fill_rate,
            execution_duration=execution_duration,
        )

        # 创建质量指标
        metrics = QualityMetrics(
            execution_id=execution_id,
            symbol=record.symbol,
            benchmark_type=benchmark_type,
            implementation_shortfall=is_value,
            is_components=is_components,
            vwap_slippage=vwap_slippage,
            arrival_price_slippage=arrival_slippage,
            fill_rate=fill_rate,
            execution_duration=execution_duration,
            num_fills=1,  # 简化版本
            average_fill_size=record.executed_quantity,
            price_impact=price_impact,
            volume_participation=volume_participation,
            quality_score=quality_score,
        )

        # 存储
        self._quality_metrics[execution_id] = metrics

        # 更新统计
        self._update_quality_stats(quality_score)

        return metrics

    def calculate_cost_breakdown(
        self,
        execution_id: str,
    ) -> CostBreakdown:
        """计算成本分解"""
        record = self.get_execution(execution_id)
        if not record:
            raise ValueError(f"Execution {execution_id} not found")

        if record.arrival_price <= 0 or record.executed_quantity <= 0:
            return CostBreakdown(
                execution_id=execution_id,
                symbol=record.symbol,
                total_cost=0.0,
                total_cost_amount=0.0,
            )

        # 计算滑点
        direction = 1 if record.side == ExecutionSide.BUY else -1
        price_diff = (record.executed_price - record.arrival_price) * direction
        slippage_bps = (price_diff / record.arrival_price) * 10000
        slippage_amount = price_diff * record.executed_quantity

        # 计算市场冲击成本
        market_impact = self._calculator.calculate_market_impact(
            record,
            record.arrival_price,
            record.executed_price,
        )

        # 计算时机成本
        timing_cost = 0.0
        if record.decision_time and record.order_time:
            # 简化：使用到达价格作为决策价格
            timing_cost = self._calculator.calculate_timing_cost(
                record,
                record.arrival_price,
                record.arrival_price,
            )

        # 计算机会成本
        opportunity_cost = self._calculator.calculate_opportunity_cost(
            record,
            record.order_quantity,
            record.closing_price if record.closing_price > 0 else record.executed_price,
        )

        # 计算价差成本
        spread_cost = self.config.spread_estimate * 10000  # 转换为bps

        # 计算佣金成本
        commission_bps = self.config.commission_rate * 10000

        # 总成本
        total_cost = slippage_bps + commission_bps
        total_cost_amount = slippage_amount + (record.executed_price * record.executed_quantity * self.config.commission_rate)

        breakdown = CostBreakdown(
            execution_id=execution_id,
            symbol=record.symbol,
            total_cost=total_cost,
            total_cost_amount=total_cost_amount,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            spread_cost=spread_cost,
            commission=commission_bps,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
        )

        # 存储
        self._cost_breakdowns[execution_id] = breakdown

        return breakdown

    # ==================== 告警检查 ====================

    def check_alerts(self, execution_id: str) -> List[ExecutionAlert]:
        """检查执行告警"""
        if not self.config.enable_alerts:
            return []

        alerts = []

        # 获取质量指标
        metrics = self._quality_metrics.get(execution_id)
        if not metrics:
            metrics = self.calculate_quality_metrics(execution_id)

        # 获取成本分解
        breakdown = self._cost_breakdowns.get(execution_id)
        if not breakdown:
            breakdown = self.calculate_cost_breakdown(execution_id)

        record = self.get_execution(execution_id)
        if not record:
            return []

        # 检查高滑点
        if abs(metrics.arrival_price_slippage) > self.config.high_slippage_threshold:
            alert = ExecutionAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.HIGH_SLIPPAGE,
                severity=AlertSeverity.CRITICAL if abs(metrics.arrival_price_slippage) > self.config.high_slippage_threshold * 2 else AlertSeverity.WARNING,
                execution_id=execution_id,
                symbol=record.symbol,
                message=f"High slippage detected: {metrics.arrival_price_slippage:.2f} bps",
                trigger_value=abs(metrics.arrival_price_slippage),
                threshold_value=self.config.high_slippage_threshold,
                recommendation="Review execution strategy and market conditions",
            )
            alerts.append(alert)

        # 检查高冲击成本
        if breakdown.market_impact > self.config.high_impact_threshold:
            alert = ExecutionAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.HIGH_IMPACT,
                severity=AlertSeverity.WARNING,
                execution_id=execution_id,
                symbol=record.symbol,
                message=f"High market impact: {breakdown.market_impact:.2f} bps",
                trigger_value=breakdown.market_impact,
                threshold_value=self.config.high_impact_threshold,
                recommendation="Consider reducing order size or using TWAP/VWAP algorithm",
            )
            alerts.append(alert)

        # 检查慢执行
        if metrics.execution_duration > self.config.slow_execution_threshold:
            alert = ExecutionAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.SLOW_EXECUTION,
                severity=AlertSeverity.INFO,
                execution_id=execution_id,
                symbol=record.symbol,
                message=f"Slow execution: {metrics.execution_duration:.0f} seconds",
                trigger_value=metrics.execution_duration,
                threshold_value=self.config.slow_execution_threshold,
                recommendation="Review execution algorithm and liquidity conditions",
            )
            alerts.append(alert)

        # 检查逆向选择
        if metrics.price_impact > self.config.adverse_selection_threshold:
            alert = ExecutionAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=AlertType.ADVERSE_SELECTION,
                severity=AlertSeverity.CRITICAL,
                execution_id=execution_id,
                symbol=record.symbol,
                message=f"Adverse selection detected: {metrics.price_impact:.2f} bps",
                trigger_value=metrics.price_impact,
                threshold_value=self.config.adverse_selection_threshold,
                recommendation="Possible information leakage, review order routing",
            )
            alerts.append(alert)

        # 存储和触发告警
        for alert in alerts:
            self._alerts.append(alert)
            self.stats['alerts_triggered'] += 1

            if self._on_alert:
                self._on_alert(alert)

        return alerts

    # ==================== 统计分析 ====================

    def calculate_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
    ) -> ExecutionStatistics:
        """计算执行统计"""
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=1))

        # 筛选执行记录
        filtered_records = []
        for record in self._executions.values():
            if start_time <= record.order_time <= end_time:
                if symbols is None or record.symbol in symbols:
                    filtered_records.append(record)

        if not filtered_records:
            return ExecutionStatistics(
                period_start=start_time,
                period_end=end_time,
            )

        # 基本统计
        total_executions = len(filtered_records)
        total_volume = sum(r.executed_quantity for r in filtered_records)
        total_value = sum(r.executed_quantity * r.executed_price for r in filtered_records)

        # 收集滑点数据
        slippages = []
        impacts = []
        fill_rates = []
        execution_times = []
        quality_scores = []

        for record in filtered_records:
            # 获取或计算质量指标
            metrics = self._quality_metrics.get(record.execution_id)
            if not metrics:
                try:
                    metrics = self.calculate_quality_metrics(record.execution_id)
                except Exception as exc:
                    # Dropping the record silently biases every aggregate below.
                    logger.warning(
                        "Quality metrics unavailable for execution %s; it is excluded "
                        "from these statistics: %s", record.execution_id, exc
                    )
                    continue

            slippages.append(abs(metrics.arrival_price_slippage))
            impacts.append(metrics.price_impact)
            fill_rates.append(metrics.fill_rate)
            execution_times.append(metrics.execution_duration)
            quality_scores.append(metrics.quality_score)

        # 计算统计值
        avg_slippage = statistics.mean(slippages) if slippages else 0.0
        median_slippage = statistics.median(slippages) if slippages else 0.0
        std_slippage = statistics.stdev(slippages) if len(slippages) > 1 else 0.0
        avg_impact = statistics.mean(impacts) if impacts else 0.0
        avg_fill_rate = statistics.mean(fill_rates) if fill_rates else 0.0
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0

        # 超过阈值的执行数
        executions_above_threshold = sum(
            1 for s in slippages
            if s > self.config.high_slippage_threshold
        )

        # 按经纪商统计
        broker_stats = self._calculate_broker_stats(filtered_records)

        # 按算法统计
        algo_stats = self._calculate_algo_stats(filtered_records)

        # 按时段统计
        hourly_stats = self._calculate_hourly_stats(filtered_records)

        return ExecutionStatistics(
            period_start=start_time,
            period_end=end_time,
            total_executions=total_executions,
            total_volume=total_volume,
            total_value=total_value,
            avg_slippage_bps=avg_slippage,
            median_slippage_bps=median_slippage,
            std_slippage_bps=std_slippage,
            avg_impact_bps=avg_impact,
            avg_fill_rate=avg_fill_rate,
            avg_execution_time=avg_execution_time,
            avg_quality_score=avg_quality_score,
            executions_above_threshold=executions_above_threshold,
            broker_stats=broker_stats,
            algo_stats=algo_stats,
            hourly_stats=hourly_stats,
        )

    def _calculate_broker_stats(self, records: List[ExecutionRecord]) -> Dict[str, Dict[str, float]]:
        """按经纪商统计"""
        broker_data = defaultdict(list)

        for record in records:
            if record.broker:
                metrics = self._quality_metrics.get(record.execution_id)
                if metrics:
                    broker_data[record.broker].append(metrics)

        broker_stats = {}
        for broker, metrics_list in broker_data.items():
            broker_stats[broker] = {
                'count': len(metrics_list),
                'avg_slippage': statistics.mean(m.arrival_price_slippage for m in metrics_list),
                'avg_quality_score': statistics.mean(m.quality_score for m in metrics_list),
            }

        return broker_stats

    def _calculate_algo_stats(self, records: List[ExecutionRecord]) -> Dict[str, Dict[str, float]]:
        """按算法统计"""
        algo_data = defaultdict(list)

        for record in records:
            if record.algo_name:
                metrics = self._quality_metrics.get(record.execution_id)
                if metrics:
                    algo_data[record.algo_name].append(metrics)

        algo_stats = {}
        for algo, metrics_list in algo_data.items():
            algo_stats[algo] = {
                'count': len(metrics_list),
                'avg_slippage': statistics.mean(m.arrival_price_slippage for m in metrics_list),
                'avg_quality_score': statistics.mean(m.quality_score for m in metrics_list),
            }

        return algo_stats

    def _calculate_hourly_stats(self, records: List[ExecutionRecord]) -> Dict[int, Dict[str, float]]:
        """按时段统计"""
        hourly_data = defaultdict(list)

        for record in records:
            hour = record.order_time.hour
            metrics = self._quality_metrics.get(record.execution_id)
            if metrics:
                hourly_data[hour].append(metrics)

        hourly_stats = {}
        for hour, metrics_list in hourly_data.items():
            hourly_stats[hour] = {
                'count': len(metrics_list),
                'avg_slippage': statistics.mean(m.arrival_price_slippage for m in metrics_list),
                'avg_quality_score': statistics.mean(m.quality_score for m in metrics_list),
            }

        return hourly_stats

    def _update_quality_stats(self, quality_score: float):
        """更新质量统计"""
        current_avg = self.stats['avg_quality_score']
        total = self.stats['total_executions']

        if total > 0:
            self.stats['avg_quality_score'] = (current_avg * (total - 1) + quality_score) / total
        else:
            self.stats['avg_quality_score'] = quality_score

    # ==================== 监控控制 ====================

    def on_alert(self, callback: Callable[[ExecutionAlert], None]):
        """注册告警回调"""
        self._on_alert = callback

    def on_execution_complete(self, callback: Callable[[ExecutionRecord, QualityMetrics], None]):
        """注册执行完成回调"""
        self._on_execution_complete = callback

    async def start_monitoring(self):
        """启动监控"""
        if self._state == MonitorState.ACTIVE:
            return

        self._state = MonitorState.ACTIVE
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Execution quality monitoring started")

    async def stop_monitoring(self):
        """停止监控"""
        self._state = MonitorState.IDLE
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Execution quality monitoring stopped")

    async def _monitor_loop(self):
        """监控循环"""
        while self._state == MonitorState.ACTIVE:
            try:
                # 检查待处理的执行
                for execution_id, record in list(self._executions.items()):
                    if record.status == ExecutionStatus.FILLED:
                        # 检查是否已计算质量指标
                        if execution_id not in self._quality_metrics:
                            metrics = self.calculate_quality_metrics(execution_id)
                            self.check_alerts(execution_id)

                            if self._on_execution_complete:
                                self._on_execution_complete(record, metrics)

                await asyncio.sleep(self.config.monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = MonitorState.ERROR
                await asyncio.sleep(5)

    # ==================== 查询方法 ====================

    def get_quality_metrics(self, execution_id: str) -> Optional[QualityMetrics]:
        """获取质量指标"""
        return self._quality_metrics.get(execution_id)

    def get_cost_breakdown(self, execution_id: str) -> Optional[CostBreakdown]:
        """获取成本分解"""
        return self._cost_breakdowns.get(execution_id)

    def get_alerts(self, limit: int = 100) -> List[ExecutionAlert]:
        """获取告警列表"""
        return list(self._alerts)[-limit:]

    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'state': self._state.value,
            'total_executions': self.stats['total_executions'],
            'alerts_triggered': self.stats['alerts_triggered'],
            'avg_quality_score': self.stats['avg_quality_score'],
            'tracked_symbols': len(self._executions_by_symbol),
            'config': self.config.to_dict(),
        }

    @property
    def state(self) -> MonitorState:
        """获取监控器状态"""
        return self._state


# ==================== 工厂函数 ====================

def create_execution_monitor(
    high_slippage_threshold: float = 10.0,
    high_impact_threshold: float = 15.0,
    enable_alerts: bool = True,
    **kwargs
) -> ExecutionQualityMonitor:
    """
    创建执行质量监控器

    Args:
        high_slippage_threshold: 高滑点阈值(bps)
        high_impact_threshold: 高冲击阈值(bps)
        enable_alerts: 启用告警
        **kwargs: 其他配置参数
    """
    config = MonitorConfig(
        high_slippage_threshold=high_slippage_threshold,
        high_impact_threshold=high_impact_threshold,
        enable_alerts=enable_alerts,
        **{k: v for k, v in kwargs.items() if hasattr(MonitorConfig, k)}
    )

    return ExecutionQualityMonitor(config)


def create_execution_record(
    symbol: str,
    side: str,
    order_quantity: float,
    arrival_price: float,
    **kwargs
) -> ExecutionRecord:
    """创建执行记录"""
    try:
        exec_side = ExecutionSide(side.lower())
    except ValueError:
        exec_side = ExecutionSide.BUY

    return ExecutionRecord(
        execution_id=str(uuid.uuid4()),
        symbol=symbol,
        side=exec_side,
        order_id=kwargs.get('order_id', str(uuid.uuid4())),
        order_quantity=order_quantity,
        order_time=kwargs.get('order_time', datetime.now()),
        arrival_price=arrival_price,
        decision_time=kwargs.get('decision_time', datetime.now()),
        **{k: v for k, v in kwargs.items() if k in ExecutionRecord.__dataclass_fields__}
    )


def quick_quality_analysis(
    symbol: str,
    side: str,
    order_quantity: float,
    executed_quantity: float,
    executed_price: float,
    arrival_price: float,
    **kwargs
) -> Dict[str, Any]:
    """
    快速质量分析

    Args:
        symbol: 交易品种
        side: 方向(buy/sell)
        order_quantity: 订单数量
        executed_quantity: 执行数量
        executed_price: 执行价格
        arrival_price: 到达价格
        **kwargs: 其他参数
    """
    monitor = create_execution_monitor()

    # 创建执行记录
    record = create_execution_record(
        symbol=symbol,
        side=side,
        order_quantity=order_quantity,
        arrival_price=arrival_price,
        executed_quantity=executed_quantity,
        executed_price=executed_price,
        status=ExecutionStatus.FILLED,
        execution_time=datetime.now(),
        **kwargs
    )

    # 添加到监控器
    execution_id = monitor.add_execution(record)

    # 计算质量指标
    metrics = monitor.calculate_quality_metrics(execution_id)

    # 计算成本分解
    breakdown = monitor.calculate_cost_breakdown(execution_id)

    # 检查告警
    alerts = monitor.check_alerts(execution_id)

    return {
        'symbol': symbol,
        'execution_id': execution_id,
        'quality_metrics': metrics.to_dict(),
        'cost_breakdown': breakdown.to_dict(),
        'alerts': [a.to_dict() for a in alerts],
        'statistics': monitor.get_statistics_summary(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'BenchmarkType',
    'ExecutionSide',
    'ExecutionStatus',
    'AlertType',
    'AlertSeverity',
    'MonitorState',
    # 数据类
    'ExecutionRecord',
    'CostBreakdown',
    'QualityMetrics',
    'ExecutionAlert',
    'ExecutionStatistics',
    'MonitorConfig',
    # 计算器
    'QualityCalculator',
    # 核心监控器
    'ExecutionQualityMonitor',
    # 工厂函数
    'create_execution_monitor',
    'create_execution_record',
    'quick_quality_analysis',
]
