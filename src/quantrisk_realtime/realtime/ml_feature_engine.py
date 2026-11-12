"""
ML Feature Engine - 机器学习特征引擎

实时特征工程、特征选择、特征变换和特征监控系统。
支持自动特征生成、特征存储和特征质量监控。

核心功能：
1. 自动特征工程 - 技术指标特征、滞后特征、滚动统计特征、交互特征
2. 特征选择 - 相关性过滤、重要性排序、递归特征消除、前向/后向选择
3. 特征变换 - 标准化、对数变换、Box-Cox变换、分位数变换
4. 特征存储与版本管理 - 特征注册表、版本控制、血缘追踪
5. 实时特征计算 - 流式特征、窗口特征、事件驱动特征、特征缓存
6. 特征监控 - 特征漂移检测、特征质量指标、缺失值追踪、异常值检测
"""

import asyncio
import logging
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
import numpy as np
from scipy import stats
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============


class FeatureType(str, Enum):
    """特征类型"""
    TECHNICAL = "technical"           # 技术指标特征
    LAG = "lag"                       # 滞后特征
    ROLLING = "rolling"               # 滚动统计特征
    INTERACTION = "interaction"       # 交互特征
    CUSTOM = "custom"                 # 自定义特征
    TRANSFORMED = "transformed"       # 变换特征


class TransformMethod(str, Enum):
    """变换方法"""
    STANDARDIZE = "standardize"       # 标准化（Z-score）
    NORMALIZE = "normalize"           # 归一化（Min-Max）
    LOG = "log"                       # 对数变换
    SQRT = "sqrt"                     # 平方根变换
    BOX_COX = "box_cox"              # Box-Cox变换
    QUANTILE = "quantile"            # 分位数变换
    RANK = "rank"                    # 排名变换


class SelectionMethod(str, Enum):
    """特征选择方法"""
    CORRELATION = "correlation"       # 相关性过滤
    VARIANCE = "variance"             # 方差过滤
    MUTUAL_INFO = "mutual_info"      # 互信息
    CHI_SQUARE = "chi_square"        # 卡方检验
    IMPORTANCE = "importance"         # 重要性排序
    RFE = "rfe"                      # 递归特征消除
    FORWARD = "forward"              # 前向选择
    BACKWARD = "backward"            # 后向选择


class DriftType(str, Enum):
    """漂移类型"""
    MEAN_SHIFT = "mean_shift"        # 均值漂移
    VARIANCE_SHIFT = "variance_shift" # 方差漂移
    DISTRIBUTION_SHIFT = "distribution_shift"  # 分布漂移
    CORRELATION_SHIFT = "correlation_shift"    # 相关性漂移


class AlertSeverity(str, Enum):
    """警报严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class CacheStrategy(str, Enum):
    """缓存策略"""
    NO_CACHE = "no_cache"
    LRU = "lru"
    TTL = "ttl"
    ALWAYS = "always"


# ============ Data Classes ============


@dataclass
class FeatureMetadata:
    """特征元数据"""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    description: str
    dependencies: List[str]          # 依赖的特征或数据
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "feature_type": self.feature_type.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "properties": self.properties,
        }


@dataclass
class FeatureValue:
    """特征值"""
    feature_id: str
    value: float
    timestamp: datetime
    confidence: float = 1.0          # 置信度
    is_valid: bool = True            # 是否有效
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


@dataclass
class FeatureVector:
    """特征向量"""
    symbol: str
    features: Dict[str, FeatureValue]
    timestamp: datetime
    is_complete: bool = True         # 是否完整（无缺失值）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "timestamp": self.timestamp.isoformat(),
            "is_complete": self.is_complete,
        }

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """转换为数组"""
        values = []
        for name in feature_names:
            if name in self.features:
                values.append(self.features[name].value)
            else:
                values.append(np.nan)
        return np.array(values)


@dataclass
class FeatureStatistics:
    """特征统计信息"""
    feature_id: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    q25: float                       # 25%分位数
    q50: float                       # 中位数
    q75: float                       # 75%分位数
    missing_rate: float              # 缺失率
    outlier_rate: float              # 异常值率
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "q25": self.q25,
            "q50": self.q50,
            "q75": self.q75,
            "missing_rate": self.missing_rate,
            "outlier_rate": self.outlier_rate,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeatureDrift:
    """特征漂移"""
    feature_id: str
    drift_type: DriftType
    drift_score: float               # 漂移分数（0-1）
    baseline_stats: FeatureStatistics
    current_stats: FeatureStatistics
    is_drifted: bool                 # 是否发生漂移
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "drift_type": self.drift_type.value,
            "drift_score": self.drift_score,
            "baseline_stats": self.baseline_stats.to_dict(),
            "current_stats": self.current_stats.to_dict(),
            "is_drifted": self.is_drifted,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_id: str
    importance_score: float
    rank: int
    method: str                      # 重要性计算方法
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "importance_score": self.importance_score,
            "rank": self.rank,
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeatureAlert:
    """特征警报"""
    alert_id: str
    feature_id: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "feature_id": self.feature_id,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TransformConfig:
    """变换配置"""
    method: TransformMethod
    params: Dict[str, Any] = field(default_factory=dict)
    fit_data: Optional[Dict[str, Any]] = None  # 拟合数据（如均值、方差）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "params": self.params,
            "fit_data": self.fit_data,
        }


@dataclass
class SelectionResult:
    """特征选择结果"""
    method: SelectionMethod
    selected_features: List[str]
    feature_scores: Dict[str, float]
    feature_ranks: Dict[str, int]
    selection_threshold: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "selected_features": self.selected_features,
            "feature_scores": self.feature_scores,
            "feature_ranks": self.feature_ranks,
            "selection_threshold": self.selection_threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EngineConfig:
    """引擎配置"""
    # 特征生成
    enable_technical_features: bool = True
    enable_lag_features: bool = True
    enable_rolling_features: bool = True
    enable_interaction_features: bool = True
    # 特征选择
    auto_feature_selection: bool = False
    max_features: int = 100
    correlation_threshold: float = 0.95
    # 特征变换
    auto_transform: bool = False
    default_transform: TransformMethod = TransformMethod.STANDARDIZE
    # 特征监控
    enable_drift_detection: bool = True
    drift_detection_window: int = 100
    drift_threshold: float = 0.1
    # 缓存
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_size: int = 1000
    cache_ttl_seconds: int = 300
    # 更新
    update_interval_ms: int = 1000
    history_size: int = 1000


# ============ Abstract Interfaces ============


class DataProvider(ABC):
    """数据提供者接口"""

    @abstractmethod
    async def get_price_data(self, symbol: str, window: int) -> List[float]:
        """获取价格数据"""
        pass

    @abstractmethod
    async def get_volume_data(self, symbol: str, window: int) -> List[float]:
        """获取成交量数据"""
        pass

    @abstractmethod
    async def get_features(self, symbol: str) -> Dict[str, float]:
        """获取现有特征"""
        pass


class FeatureStorage(ABC):
    """特征存储接口"""

    @abstractmethod
    async def save_feature_vector(self, vector: FeatureVector) -> None:
        """保存特征向量"""
        pass

    @abstractmethod
    async def load_feature_vector(
        self, symbol: str, timestamp: datetime
    ) -> Optional[FeatureVector]:
        """加载特征向量"""
        pass

    @abstractmethod
    async def save_metadata(self, metadata: FeatureMetadata) -> None:
        """保存特征元数据"""
        pass

    @abstractmethod
    async def load_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """加载特征元数据"""
        pass


# ============ Simulated Providers ============


class SimulatedDataProvider(DataProvider):
    """模拟数据提供者"""

    def __init__(self):
        self._price_data: Dict[str, List[float]] = {}
        self._volume_data: Dict[str, List[float]] = {}
        self._features: Dict[str, Dict[str, float]] = {}

    def set_price_data(self, symbol: str, data: List[float]) -> None:
        self._price_data[symbol] = data

    def set_volume_data(self, symbol: str, data: List[float]) -> None:
        self._volume_data[symbol] = data

    def set_features(self, symbol: str, features: Dict[str, float]) -> None:
        self._features[symbol] = features

    async def get_price_data(self, symbol: str, window: int) -> List[float]:
        data = self._price_data.get(symbol, [])
        return data[-window:] if data else []

    async def get_volume_data(self, symbol: str, window: int) -> List[float]:
        data = self._volume_data.get(symbol, [])
        return data[-window:] if data else []

    async def get_features(self, symbol: str) -> Dict[str, float]:
        return self._features.get(symbol, {})


class InMemoryFeatureStorage(FeatureStorage):
    """内存特征存储"""

    def __init__(self):
        self._vectors: Dict[str, List[FeatureVector]] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}

    async def save_feature_vector(self, vector: FeatureVector) -> None:
        if vector.symbol not in self._vectors:
            self._vectors[vector.symbol] = []
        self._vectors[vector.symbol].append(vector)

    async def load_feature_vector(
        self, symbol: str, timestamp: datetime
    ) -> Optional[FeatureVector]:
        vectors = self._vectors.get(symbol, [])
        for vector in reversed(vectors):
            if vector.timestamp <= timestamp:
                return vector
        return None

    async def save_metadata(self, metadata: FeatureMetadata) -> None:
        self._metadata[metadata.feature_id] = metadata

    async def load_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        return self._metadata.get(feature_id)


# ============ Feature Generators ============


class TechnicalFeatureGenerator:
    """技术指标特征生成器"""

    @staticmethod
    def generate_returns(prices: List[float], periods: List[int] = None) -> Dict[str, float]:
        """生成收益率特征"""
        if not prices or len(prices) < 2:
            return {}

        if periods is None:
            periods = [1, 5, 10, 20]

        features = {}
        for period in periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1]
                features[f"return_{period}d"] = ret

        return features

    @staticmethod
    def generate_moving_averages(prices: List[float], windows: List[int] = None) -> Dict[str, float]:
        """生成移动平均特征"""
        if not prices:
            return {}

        if windows is None:
            windows = [5, 10, 20, 50]

        features = {}
        for window in windows:
            if len(prices) >= window:
                ma = sum(prices[-window:]) / window
                features[f"ma_{window}"] = ma
                # MA比率
                features[f"price_to_ma_{window}"] = prices[-1] / ma - 1.0

        return features

    @staticmethod
    def generate_volatility(prices: List[float], windows: List[int] = None) -> Dict[str, float]:
        """生成波动率特征"""
        if not prices or len(prices) < 2:
            return {}

        if windows is None:
            windows = [5, 10, 20]

        features = {}
        for window in windows:
            if len(prices) > window:
                returns = [
                    (prices[i] - prices[i-1]) / prices[i-1]
                    for i in range(len(prices)-window, len(prices))
                ]
                vol = np.std(returns) * np.sqrt(252)  # 年化波动率
                features[f"volatility_{window}d"] = vol

        return features

    @staticmethod
    def generate_momentum(prices: List[float]) -> Dict[str, float]:
        """生成动量特征"""
        if not prices or len(prices) < 20:
            return {}

        features = {}

        # RSI
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, 15):
                change = prices[-i] - prices[-i-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))

            avg_gain = sum(gains) / 14 if gains else 0
            avg_loss = sum(losses) / 14 if losses else 0

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features["rsi_14"] = rsi

        # MACD
        if len(prices) >= 26:
            ema_12 = TechnicalFeatureGenerator._ema(prices, 12)
            ema_26 = TechnicalFeatureGenerator._ema(prices, 26)
            features["macd"] = ema_12 - ema_26

        return features

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """计算指数移动平均"""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = (price - ema) * multiplier + ema

        return ema


class LagFeatureGenerator:
    """滞后特征生成器"""

    @staticmethod
    def generate(data: List[float], lags: List[int] = None) -> Dict[str, float]:
        """生成滞后特征"""
        if not data:
            return {}

        if lags is None:
            lags = [1, 2, 3, 5, 10]

        features = {}
        for lag in lags:
            if len(data) > lag:
                features[f"lag_{lag}"] = data[-lag-1]

        return features


class RollingFeatureGenerator:
    """滚动统计特征生成器"""

    @staticmethod
    def generate(data: List[float], windows: List[int] = None) -> Dict[str, float]:
        """生成滚动统计特征"""
        if not data:
            return {}

        if windows is None:
            windows = [5, 10, 20]

        features = {}
        for window in windows:
            if len(data) >= window:
                window_data = data[-window:]

                # 均值
                features[f"rolling_mean_{window}"] = np.mean(window_data)

                # 标准差
                features[f"rolling_std_{window}"] = np.std(window_data)

                # 最小值
                features[f"rolling_min_{window}"] = np.min(window_data)

                # 最大值
                features[f"rolling_max_{window}"] = np.max(window_data)

                # 偏度
                if len(window_data) > 2:
                    features[f"rolling_skew_{window}"] = stats.skew(window_data)

                # 峰度
                if len(window_data) > 3:
                    features[f"rolling_kurt_{window}"] = stats.kurtosis(window_data)

        return features


class InteractionFeatureGenerator:
    """交互特征生成器"""

    @staticmethod
    def generate(features: Dict[str, float], pairs: List[Tuple[str, str]] = None) -> Dict[str, float]:
        """生成交互特征"""
        if not features:
            return {}

        interaction_features = {}

        if pairs is None:
            # 自动生成所有两两组合
            feature_names = list(features.keys())
            pairs = [
                (feature_names[i], feature_names[j])
                for i in range(len(feature_names))
                for j in range(i+1, len(feature_names))
            ]

        for f1, f2 in pairs:
            if f1 in features and f2 in features:
                v1, v2 = features[f1], features[f2]

                # 乘积
                interaction_features[f"{f1}_x_{f2}"] = v1 * v2

                # 比率（避免除零）
                if abs(v2) > 1e-8:
                    interaction_features[f"{f1}_div_{f2}"] = v1 / v2

                # 差值
                interaction_features[f"{f1}_minus_{f2}"] = v1 - v2

        return interaction_features


# ============ Feature Transformers ============


class FeatureTransformer:
    """特征变换器"""

    def __init__(self):
        self._configs: Dict[str, TransformConfig] = {}

    def fit(self, feature_id: str, values: List[float], method: TransformMethod) -> None:
        """拟合变换参数"""
        if not values:
            return

        fit_data = {}

        if method == TransformMethod.STANDARDIZE:
            fit_data["mean"] = np.mean(values)
            fit_data["std"] = np.std(values)

        elif method == TransformMethod.NORMALIZE:
            fit_data["min"] = np.min(values)
            fit_data["max"] = np.max(values)

        elif method == TransformMethod.BOX_COX:
            # 确保所有值为正
            if np.min(values) > 0:
                fitted_data, lambda_param = stats.boxcox(values)
                fit_data["lambda"] = lambda_param
            else:
                fit_data["lambda"] = 1.0

        elif method == TransformMethod.QUANTILE:
            fit_data["quantiles"] = np.percentile(values, np.linspace(0, 100, 100)).tolist()

        self._configs[feature_id] = TransformConfig(
            method=method,
            fit_data=fit_data
        )

    def transform(self, feature_id: str, value: float) -> float:
        """变换单个值"""
        if feature_id not in self._configs:
            return value

        config = self._configs[feature_id]
        method = config.method
        fit_data = config.fit_data or {}

        try:
            if method == TransformMethod.STANDARDIZE:
                mean = fit_data.get("mean", 0)
                std = fit_data.get("std", 1)
                return (value - mean) / std if std > 0 else 0

            elif method == TransformMethod.NORMALIZE:
                min_val = fit_data.get("min", 0)
                max_val = fit_data.get("max", 1)
                range_val = max_val - min_val
                return (value - min_val) / range_val if range_val > 0 else 0

            elif method == TransformMethod.LOG:
                return np.log(value) if value > 0 else 0

            elif method == TransformMethod.SQRT:
                return np.sqrt(value) if value >= 0 else 0

            elif method == TransformMethod.BOX_COX:
                lambda_param = fit_data.get("lambda", 1.0)
                if value > 0:
                    if abs(lambda_param) < 1e-8:
                        return np.log(value)
                    else:
                        return (value ** lambda_param - 1) / lambda_param
                return 0

            elif method == TransformMethod.QUANTILE:
                quantiles = fit_data.get("quantiles", [])
                if quantiles:
                    # 找到值所在的分位数
                    rank = np.searchsorted(quantiles, value)
                    return rank / len(quantiles)
                return 0

            elif method == TransformMethod.RANK:
                # 简化实现，需要批量数据才能准确排名
                return value

        except Exception as e:
            logger.error(f"Transform error for {feature_id}: {e}")
            return value

        return value


# ============ Feature Selectors ============


class FeatureSelector:
    """特征选择器"""

    @staticmethod
    def select_by_correlation(
        features: Dict[str, List[float]],
        threshold: float = 0.95
    ) -> List[str]:
        """基于相关性过滤特征"""
        if not features:
            return []

        feature_names = list(features.keys())
        n_features = len(feature_names)

        if n_features <= 1:
            return feature_names

        # 计算相关矩阵
        feature_matrix = np.array([features[name] for name in feature_names]).T
        corr_matrix = np.corrcoef(feature_matrix, rowvar=False)

        # 找出高度相关的特征对
        to_remove = set()
        for i in range(n_features):
            if feature_names[i] in to_remove:
                continue
            for j in range(i+1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    to_remove.add(feature_names[j])

        return [name for name in feature_names if name not in to_remove]

    @staticmethod
    def select_by_variance(
        features: Dict[str, List[float]],
        threshold: float = 0.01
    ) -> List[str]:
        """基于方差过滤特征"""
        selected = []
        for name, values in features.items():
            if np.var(values) > threshold:
                selected.append(name)
        return selected

    @staticmethod
    def select_by_importance(
        importances: Dict[str, float],
        top_k: int = None,
        threshold: float = None
    ) -> List[str]:
        """基于重要性选择特征"""
        if not importances:
            return []

        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if top_k is not None:
            return [name for name, _ in sorted_features[:top_k]]

        if threshold is not None:
            return [name for name, score in sorted_features if score >= threshold]

        return [name for name, _ in sorted_features]


# ============ Feature Monitors ============


class DriftDetector:
    """漂移检测器"""

    @staticmethod
    def detect_mean_shift(
        baseline: FeatureStatistics,
        current: FeatureStatistics,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """检测均值漂移"""
        if baseline.std == 0:
            return False, 0.0

        # 标准化差异
        diff = abs(current.mean - baseline.mean) / baseline.std
        is_drifted = diff > threshold

        return is_drifted, diff

    @staticmethod
    def detect_variance_shift(
        baseline: FeatureStatistics,
        current: FeatureStatistics,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """检测方差漂移"""
        if baseline.std == 0:
            return False, 0.0

        # 方差比率
        ratio = current.std / baseline.std
        diff = abs(ratio - 1.0)
        is_drifted = diff > threshold

        return is_drifted, diff

    @staticmethod
    def detect_distribution_shift(
        baseline_values: List[float],
        current_values: List[float],
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """检测分布漂移（KS检验）"""
        if len(baseline_values) < 2 or len(current_values) < 2:
            return False, 0.0

        try:
            statistic, p_value = stats.ks_2samp(baseline_values, current_values)
            is_drifted = p_value < threshold
            return is_drifted, statistic
        except Exception as e:
            logger.error(f"KS test error: {e}")
            return False, 0.0


# ============ Main Class ============


class MLFeatureEngine:
    """机器学习特征引擎"""

    def __init__(
        self,
        config: EngineConfig,
        data_provider: Optional[DataProvider] = None,
        feature_storage: Optional[FeatureStorage] = None,
    ):
        self._config = config
        self._data_provider = data_provider or simulated_default(SimulatedDataProvider, "data from MLFeatureEngine's simulated data_provider (SimulatedDataProvider)")
        self._feature_storage = feature_storage or InMemoryFeatureStorage()

        # 特征生成器
        self._technical_gen = TechnicalFeatureGenerator()
        self._lag_gen = LagFeatureGenerator()
        self._rolling_gen = RollingFeatureGenerator()
        self._interaction_gen = InteractionFeatureGenerator()

        # 特征变换器和选择器
        self._transformer = FeatureTransformer()
        self._selector = FeatureSelector()

        # 状态
        self._state = EngineState.IDLE
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 特征注册表
        self._registry: Dict[str, FeatureMetadata] = {}
        self._feature_history: Dict[str, deque] = {}

        # 统计信息
        self._statistics: Dict[str, FeatureStatistics] = {}
        self._baseline_statistics: Dict[str, FeatureStatistics] = {}

        # 重要性
        self._importances: Dict[str, FeatureImportance] = {}

        # 缓存
        self._cache: Dict[str, Tuple[FeatureVector, datetime]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 警报
        self._alerts: List[FeatureAlert] = []
        self._alert_counter = 0

        # 回调
        self._feature_callbacks: List[Callable[[FeatureVector], None]] = []
        self._alert_callbacks: List[Callable[[FeatureAlert], None]] = []
        self._drift_callbacks: List[Callable[[FeatureDrift], None]] = []

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动引擎"""
        if self._running:
            return

        self._running = True
        self._state = EngineState.RUNNING
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("MLFeatureEngine started")

    async def stop(self) -> None:
        """停止引擎"""
        self._running = False
        self._state = EngineState.IDLE
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("MLFeatureEngine stopped")

    async def _update_loop(self) -> None:
        """更新循环"""
        while self._running:
            try:
                # 定期清理缓存
                await self._cleanup_cache()

                # 定期检测漂移
                if self._config.enable_drift_detection:
                    await self._check_drift_all()

                await asyncio.sleep(self._config.update_interval_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1)

    # ============ Feature Generation ============

    async def generate_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> FeatureVector:
        """生成特征向量"""
        if timestamp is None:
            timestamp = datetime.now()

        # 检查缓存
        cached = self._get_from_cache(symbol)
        if cached:
            return cached

        features: Dict[str, FeatureValue] = {}

        try:
            # 获取基础数据
            prices = await self._data_provider.get_price_data(symbol, window=100)
            volumes = await self._data_provider.get_volume_data(symbol, window=100)
            existing_features = await self._data_provider.get_features(symbol)

            # 1. 技术指标特征
            if self._config.enable_technical_features and prices:
                tech_features = {}
                tech_features.update(self._technical_gen.generate_returns(prices))
                tech_features.update(self._technical_gen.generate_moving_averages(prices))
                tech_features.update(self._technical_gen.generate_volatility(prices))
                tech_features.update(self._technical_gen.generate_momentum(prices))

                for name, value in tech_features.items():
                    feature_id = f"tech_{name}"
                    features[feature_id] = FeatureValue(
                        feature_id=feature_id,
                        value=value,
                        timestamp=timestamp
                    )

            # 2. 滞后特征
            if self._config.enable_lag_features and prices:
                lag_features = self._lag_gen.generate(prices)
                for name, value in lag_features.items():
                    feature_id = f"lag_{name}"
                    features[feature_id] = FeatureValue(
                        feature_id=feature_id,
                        value=value,
                        timestamp=timestamp
                    )

            # 3. 滚动统计特征
            if self._config.enable_rolling_features and prices:
                rolling_features = self._rolling_gen.generate(prices)
                for name, value in rolling_features.items():
                    feature_id = f"rolling_{name}"
                    features[feature_id] = FeatureValue(
                        feature_id=feature_id,
                        value=value,
                        timestamp=timestamp
                    )

            # 4. 交互特征
            if self._config.enable_interaction_features:
                base_features = {k: v.value for k, v in features.items()}
                if len(base_features) >= 2:
                    # 限制交互特征数量
                    top_features = list(base_features.keys())[:10]
                    pairs = [(top_features[i], top_features[j])
                            for i in range(len(top_features))
                            for j in range(i+1, min(i+3, len(top_features)))]

                    interaction_features = self._interaction_gen.generate(base_features, pairs)
                    for name, value in interaction_features.items():
                        feature_id = f"interaction_{name}"
                        features[feature_id] = FeatureValue(
                            feature_id=feature_id,
                            value=value,
                            timestamp=timestamp
                        )

            # 5. 特征变换
            if self._config.auto_transform:
                transformed_features = {}
                for feature_id, feature_value in features.items():
                    transformed_value = self._transformer.transform(
                        feature_id,
                        feature_value.value
                    )
                    transformed_id = f"transformed_{feature_id}"
                    transformed_features[transformed_id] = FeatureValue(
                        feature_id=transformed_id,
                        value=transformed_value,
                        timestamp=timestamp
                    )
                features.update(transformed_features)

            # 创建特征向量
            vector = FeatureVector(
                symbol=symbol,
                features=features,
                timestamp=timestamp,
                is_complete=True
            )

            # 更新历史
            self._add_to_history(symbol, vector)

            # 更新统计
            await self._update_statistics(vector)

            # 缓存
            self._add_to_cache(symbol, vector)

            # 保存到存储
            await self._feature_storage.save_feature_vector(vector)

            # 通知回调
            self._notify_features(vector)

            return vector

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            return FeatureVector(
                symbol=symbol,
                features={},
                timestamp=timestamp,
                is_complete=False
            )

    # ============ Feature Registration ============

    def register_feature(self, metadata: FeatureMetadata) -> None:
        """注册特征"""
        self._registry[metadata.feature_id] = metadata
        logger.info(f"Feature registered: {metadata.feature_id}")

    def get_feature_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """获取特征元数据"""
        return self._registry.get(feature_id)

    def list_features(self) -> List[FeatureMetadata]:
        """列出所有特征"""
        return list(self._registry.values())

    # ============ Feature Transformation ============

    def fit_transformer(
        self,
        feature_id: str,
        values: List[float],
        method: TransformMethod
    ) -> None:
        """拟合变换器"""
        self._transformer.fit(feature_id, values, method)

    def transform_feature(self, feature_id: str, value: float) -> float:
        """变换特征值"""
        return self._transformer.transform(feature_id, value)

    # ============ Feature Selection ============

    async def select_features(
        self,
        method: SelectionMethod,
        feature_data: Dict[str, List[float]],
        **kwargs
    ) -> SelectionResult:
        """选择特征"""
        timestamp = datetime.now()
        selected_features = []
        feature_scores = {}
        feature_ranks = {}

        if method == SelectionMethod.CORRELATION:
            threshold = kwargs.get("threshold", self._config.correlation_threshold)
            selected_features = self._selector.select_by_correlation(
                feature_data, threshold
            )
            # 简单评分
            for name in selected_features:
                feature_scores[name] = 1.0

        elif method == SelectionMethod.VARIANCE:
            threshold = kwargs.get("threshold", 0.01)
            selected_features = self._selector.select_by_variance(
                feature_data, threshold
            )
            for name in selected_features:
                feature_scores[name] = np.var(feature_data[name])

        elif method == SelectionMethod.IMPORTANCE:
            importances = {k: v.importance_score for k, v in self._importances.items()}
            top_k = kwargs.get("top_k", self._config.max_features)
            selected_features = self._selector.select_by_importance(
                importances, top_k=top_k
            )
            feature_scores = importances

        # 计算排名
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (name, _) in enumerate(sorted_features, 1):
            feature_ranks[name] = rank

        result = SelectionResult(
            method=method,
            selected_features=selected_features,
            feature_scores=feature_scores,
            feature_ranks=feature_ranks,
            selection_threshold=kwargs.get("threshold", 0.0),
            timestamp=timestamp,
        )

        return result

    # ============ Feature Importance ============

    def set_feature_importance(
        self,
        feature_id: str,
        importance_score: float,
        method: str = "manual"
    ) -> None:
        """设置特征重要性"""
        # 计算排名
        all_importances = {
            k: v.importance_score
            for k, v in self._importances.items()
        }
        all_importances[feature_id] = importance_score

        sorted_features = sorted(
            all_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        rank = next(
            (i for i, (k, _) in enumerate(sorted_features, 1) if k == feature_id),
            len(sorted_features)
        )

        self._importances[feature_id] = FeatureImportance(
            feature_id=feature_id,
            importance_score=importance_score,
            rank=rank,
            method=method,
        )

    def get_feature_importance(self, feature_id: str) -> Optional[FeatureImportance]:
        """获取特征重要性"""
        return self._importances.get(feature_id)

    def get_top_features(self, top_k: int = 10) -> List[FeatureImportance]:
        """获取最重要的特征"""
        sorted_importances = sorted(
            self._importances.values(),
            key=lambda x: x.importance_score,
            reverse=True
        )
        return sorted_importances[:top_k]

    # ============ Statistics and Monitoring ============

    async def _update_statistics(self, vector: FeatureVector) -> None:
        """更新统计信息"""
        for feature_id, feature_value in vector.features.items():
            if feature_id not in self._feature_history:
                self._feature_history[feature_id] = deque(
                    maxlen=self._config.history_size
                )

            self._feature_history[feature_id].append(feature_value.value)

            # 计算统计
            values = list(self._feature_history[feature_id])
            if len(values) >= 10:
                stats_obj = self._calculate_statistics(feature_id, values)
                self._statistics[feature_id] = stats_obj

                # 设置基线（第一次）
                if feature_id not in self._baseline_statistics:
                    self._baseline_statistics[feature_id] = stats_obj

    def _calculate_statistics(
        self,
        feature_id: str,
        values: List[float]
    ) -> FeatureStatistics:
        """计算特征统计"""
        valid_values = [v for v in values if not np.isnan(v)]
        total_count = len(values)
        valid_count = len(valid_values)

        if valid_count == 0:
            return FeatureStatistics(
                feature_id=feature_id,
                count=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                q25=0.0,
                q50=0.0,
                q75=0.0,
                missing_rate=1.0,
                outlier_rate=0.0,
            )

        mean = np.mean(valid_values)
        std = np.std(valid_values)
        q25, q50, q75 = np.percentile(valid_values, [25, 50, 75])

        # 异常值检测（IQR方法）
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = [v for v in valid_values if v < lower_bound or v > upper_bound]
        outlier_rate = len(outliers) / valid_count if valid_count > 0 else 0

        return FeatureStatistics(
            feature_id=feature_id,
            count=valid_count,
            mean=mean,
            std=std,
            min=np.min(valid_values),
            max=np.max(valid_values),
            q25=q25,
            q50=q50,
            q75=q75,
            missing_rate=(total_count - valid_count) / total_count,
            outlier_rate=outlier_rate,
        )

    async def _check_drift_all(self) -> None:
        """检查所有特征的漂移"""
        for feature_id in self._feature_history.keys():
            await self._check_drift(feature_id)

    async def _check_drift(self, feature_id: str) -> None:
        """检查特征漂移"""
        if feature_id not in self._baseline_statistics:
            return

        if feature_id not in self._statistics:
            return

        baseline = self._baseline_statistics[feature_id]
        current = self._statistics[feature_id]

        # 均值漂移
        is_mean_drifted, mean_score = DriftDetector.detect_mean_shift(
            baseline, current, self._config.drift_threshold
        )

        if is_mean_drifted:
            drift = FeatureDrift(
                feature_id=feature_id,
                drift_type=DriftType.MEAN_SHIFT,
                drift_score=mean_score,
                baseline_stats=baseline,
                current_stats=current,
                is_drifted=True,
                threshold=self._config.drift_threshold,
            )
            self._notify_drift(drift)
            self._create_alert(
                feature_id,
                AlertSeverity.MEDIUM,
                f"Mean drift detected: {mean_score:.3f}",
                {"drift_type": "mean_shift", "score": mean_score}
            )

        # 方差漂移
        is_var_drifted, var_score = DriftDetector.detect_variance_shift(
            baseline, current, self._config.drift_threshold * 2
        )

        if is_var_drifted:
            drift = FeatureDrift(
                feature_id=feature_id,
                drift_type=DriftType.VARIANCE_SHIFT,
                drift_score=var_score,
                baseline_stats=baseline,
                current_stats=current,
                is_drifted=True,
                threshold=self._config.drift_threshold * 2,
            )
            self._notify_drift(drift)

    def get_feature_statistics(self, feature_id: str) -> Optional[FeatureStatistics]:
        """获取特征统计"""
        return self._statistics.get(feature_id)

    # ============ Cache Management ============

    def _add_to_cache(self, symbol: str, vector: FeatureVector) -> None:
        """添加到缓存"""
        if self._config.cache_strategy == CacheStrategy.NO_CACHE:
            return

        self._cache[symbol] = (vector, datetime.now())

        # LRU策略：限制大小
        if len(self._cache) > self._config.cache_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]

    def _get_from_cache(self, symbol: str) -> Optional[FeatureVector]:
        """从缓存获取"""
        if self._config.cache_strategy == CacheStrategy.NO_CACHE:
            self._cache_misses += 1
            return None

        if symbol not in self._cache:
            self._cache_misses += 1
            return None

        vector, cached_time = self._cache[symbol]

        # TTL检查
        if self._config.cache_strategy == CacheStrategy.TTL:
            age = (datetime.now() - cached_time).total_seconds()
            if age > self._config.cache_ttl_seconds:
                del self._cache[symbol]
                self._cache_misses += 1
                return None

        self._cache_hits += 1
        return vector

    async def _cleanup_cache(self) -> None:
        """清理缓存"""
        if self._config.cache_strategy == CacheStrategy.TTL:
            now = datetime.now()
            expired = [
                symbol for symbol, (_, cached_time) in self._cache.items()
                if (now - cached_time).total_seconds() > self._config.cache_ttl_seconds
            ]
            for symbol in expired:
                del self._cache[symbol]

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # ============ History Management ============

    def _add_to_history(self, symbol: str, vector: FeatureVector) -> None:
        """添加到历史"""
        for feature_id, feature_value in vector.features.items():
            if feature_id not in self._feature_history:
                self._feature_history[feature_id] = deque(
                    maxlen=self._config.history_size
                )
            self._feature_history[feature_id].append(feature_value.value)

    def get_feature_history(
        self,
        feature_id: str,
        limit: int = 100
    ) -> List[float]:
        """获取特征历史"""
        if feature_id not in self._feature_history:
            return []
        history = list(self._feature_history[feature_id])
        return history[-limit:]

    # ============ Alerts ============

    def _create_alert(
        self,
        feature_id: str,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """创建警报"""
        self._alert_counter += 1
        alert = FeatureAlert(
            alert_id=f"FEAT_ALERT_{self._alert_counter}",
            feature_id=feature_id,
            severity=severity,
            message=message,
            details=details,
        )
        self._alerts.append(alert)
        self._notify_alert(alert)

    def get_alerts(self, limit: int = 100) -> List[FeatureAlert]:
        """获取警报列表"""
        return self._alerts[-limit:]

    # ============ Callbacks ============

    def on_features_update(
        self, callback: Callable[[FeatureVector], None]
    ) -> None:
        """注册特征更新回调"""
        self._feature_callbacks.append(callback)

    def on_alert(
        self, callback: Callable[[FeatureAlert], None]
    ) -> None:
        """注册警报回调"""
        self._alert_callbacks.append(callback)

    def on_drift(
        self, callback: Callable[[FeatureDrift], None]
    ) -> None:
        """注册漂移检测回调"""
        self._drift_callbacks.append(callback)

    def _notify_features(self, vector: FeatureVector) -> None:
        """通知特征更新"""
        for callback in self._feature_callbacks:
            try:
                callback(vector)
            except Exception as e:
                logger.error(f"Error in features callback: {e}")

    def _notify_alert(self, alert: FeatureAlert) -> None:
        """通知警报"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _notify_drift(self, drift: FeatureDrift) -> None:
        """通知漂移检测"""
        for callback in self._drift_callbacks:
            try:
                callback(drift)
            except Exception as e:
                logger.error(f"Error in drift callback: {e}")

    # ============ Public API ============

    def get_state(self) -> EngineState:
        """获取引擎状态"""
        return self._state

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "features_registered": len(self._registry),
            "features_tracked": len(self._feature_history),
            "total_alerts": len(self._alerts),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0,
        }


# ============ Factory Functions ============


def create_feature_engine(
    enable_technical_features: bool = True,
    enable_lag_features: bool = True,
    enable_rolling_features: bool = True,
    enable_drift_detection: bool = True,
    data_provider: Optional[DataProvider] = None,
    feature_storage: Optional[FeatureStorage] = None,
) -> MLFeatureEngine:
    """创建特征引擎"""
    config = EngineConfig(
        enable_technical_features=enable_technical_features,
        enable_lag_features=enable_lag_features,
        enable_rolling_features=enable_rolling_features,
        enable_drift_detection=enable_drift_detection,
    )
    return MLFeatureEngine(
        config=config,
        data_provider=data_provider,
        feature_storage=feature_storage,
    )


def create_feature_metadata(
    feature_id: str,
    feature_name: str,
    feature_type: str,
    description: str,
    dependencies: List[str] = None,
) -> FeatureMetadata:
    """创建特征元数据"""
    return FeatureMetadata(
        feature_id=feature_id,
        feature_name=feature_name,
        feature_type=FeatureType(feature_type),
        description=description,
        dependencies=dependencies or [],
    )


async def quick_feature_generation(
    engine: MLFeatureEngine,
    symbol: str
) -> FeatureVector:
    """快速生成特征"""
    return await engine.generate_features(symbol)


# ============ Exports ============


__all__ = [
    # Enums
    "FeatureType",
    "TransformMethod",
    "SelectionMethod",
    "DriftType",
    "AlertSeverity",
    "EngineState",
    "CacheStrategy",
    # Data Classes
    "FeatureMetadata",
    "FeatureValue",
    "FeatureVector",
    "FeatureStatistics",
    "FeatureDrift",
    "FeatureImportance",
    "FeatureAlert",
    "TransformConfig",
    "SelectionResult",
    "EngineConfig",
    # Interfaces
    "DataProvider",
    "FeatureStorage",
    # Simulated Providers
    "SimulatedDataProvider",
    "InMemoryFeatureStorage",
    # Feature Generators
    "TechnicalFeatureGenerator",
    "LagFeatureGenerator",
    "RollingFeatureGenerator",
    "InteractionFeatureGenerator",
    # Feature Transformers
    "FeatureTransformer",
    # Feature Selectors
    "FeatureSelector",
    # Monitors
    "DriftDetector",
    # Main Class
    "MLFeatureEngine",
    # Factory Functions
    "create_feature_engine",
    "create_feature_metadata",
    "quick_feature_generation",
]
