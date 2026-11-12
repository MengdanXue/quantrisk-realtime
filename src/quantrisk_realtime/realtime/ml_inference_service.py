"""
ML Inference Service - 机器学习推理服务

实时机器学习模型推理服务，包括：
- 模型注册与版本管理
- 动态特征计算
- 实时推理服务
- 预测缓存
- 模型漂移监控
- 置信度校准
- 与交易执行系统集成

这是将 ML 模型与实盘交易系统集成的关键模块。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import statistics
import uuid
import json
import hashlib
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class ModelType(str, Enum):
    """模型类型"""
    CLASSIFICATION = "classification"      # 分类模型
    REGRESSION = "regression"              # 回归模型
    RANKING = "ranking"                    # 排序模型
    TIME_SERIES = "time_series"            # 时序预测
    REINFORCEMENT = "reinforcement"        # 强化学习
    ENSEMBLE = "ensemble"                  # 集成模型


class ModelStatus(str, Enum):
    """模型状态"""
    REGISTERED = "registered"              # 已注册
    LOADING = "loading"                    # 加载中
    READY = "ready"                        # 就绪
    SERVING = "serving"                    # 服务中
    PAUSED = "paused"                      # 已暂停
    DEPRECATED = "deprecated"              # 已废弃
    ERROR = "error"                        # 错误


class PredictionType(str, Enum):
    """预测类型"""
    SIGNAL = "signal"                      # 交易信号
    DIRECTION = "direction"                # 方向预测
    PRICE_TARGET = "price_target"          # 目标价
    VOLATILITY = "volatility"              # 波动率
    PROBABILITY = "probability"            # 概率
    SCORE = "score"                        # 评分
    RANKING = "ranking"                    # 排名


class SignalDirection(str, Enum):
    """信号方向"""
    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class DriftType(str, Enum):
    """漂移类型"""
    FEATURE_DRIFT = "feature_drift"        # 特征漂移
    PREDICTION_DRIFT = "prediction_drift"  # 预测漂移
    PERFORMANCE_DRIFT = "performance_drift" # 性能漂移
    CONCEPT_DRIFT = "concept_drift"        # 概念漂移


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ServiceState(str, Enum):
    """服务状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    description: str = ""
    features: List[str] = field(default_factory=list)
    target: str = ""
    framework: str = "sklearn"  # sklearn/xgboost/lightgbm/pytorch/tensorflow
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type.value,
            "version": self.version,
            "description": self.description,
            "features": self.features,
            "target": self.target,
            "framework": self.framework,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "tags": self.tags,
        }


@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    model_id: str
    version: str
    status: ModelStatus
    artifact_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    is_champion: bool = False  # 当前主力模型

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "metrics": self.metrics,
            "is_champion": self.is_champion,
        }


@dataclass
class FeatureVector:
    """特征向量"""
    symbol: str
    features: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    feature_hash: str = ""

    def __post_init__(self):
        if not self.feature_hash:
            self.feature_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """计算特征哈希"""
        content = f"{self.symbol}:{json.dumps(self.features, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "feature_hash": self.feature_hash,
        }


@dataclass
class Prediction:
    """预测结果"""
    prediction_id: str
    model_id: str
    model_version: str
    symbol: str
    prediction_type: PredictionType
    value: Union[float, str, List[float]]
    confidence: float                      # 置信度 0-1
    probabilities: Optional[Dict[str, float]] = None  # 类别概率
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0               # 推理延迟
    feature_hash: str = ""                # 输入特征哈希
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "prediction_type": self.prediction_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "feature_hash": self.feature_hash,
            "metadata": self.metadata,
        }


@dataclass
class TradingSignal:
    """交易信号"""
    signal_id: str
    symbol: str
    direction: SignalDirection
    strength: float                        # 信号强度 0-1
    confidence: float                      # 置信度 0-1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: str = "1d"               # 信号有效期
    source_model: str = ""
    source_prediction_id: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "time_horizon": self.time_horizon,
            "source_model": self.source_model,
            "source_prediction_id": self.source_prediction_id,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired,
            "metadata": self.metadata,
        }


@dataclass
class DriftAlert:
    """漂移告警"""
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    @property
    def drift_magnitude(self) -> float:
        """漂移幅度"""
        if self.baseline_value == 0:
            return abs(self.current_value)
        return abs(self.current_value - self.baseline_value) / abs(self.baseline_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "model_id": self.model_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "drift_magnitude": self.drift_magnitude,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_id: str
    model_version: str
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    last_prediction_at: Optional[datetime] = None
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "total_predictions": self.total_predictions,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "last_prediction_at": self.last_prediction_at.isoformat() if self.last_prediction_at else None,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class InferenceConfig:
    """推理服务配置"""
    cache_ttl_seconds: int = 60            # 缓存TTL
    max_cache_size: int = 10000            # 最大缓存条目
    batch_size: int = 32                   # 批量推理大小
    timeout_ms: int = 1000                 # 超时时间
    enable_caching: bool = True            # 启用缓存
    enable_drift_detection: bool = True    # 启用漂移检测
    drift_threshold: float = 0.1           # 漂移阈值
    confidence_threshold: float = 0.5      # 置信度阈值
    history_size: int = 10000              # 历史记录大小
    update_interval: float = 1.0           # 更新间隔(秒)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_size": self.max_cache_size,
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
            "enable_caching": self.enable_caching,
            "enable_drift_detection": self.enable_drift_detection,
            "drift_threshold": self.drift_threshold,
            "confidence_threshold": self.confidence_threshold,
            "history_size": self.history_size,
            "update_interval": self.update_interval,
        }


# ============ Providers ============

class FeatureProvider(ABC):
    """特征提供者接口"""

    @abstractmethod
    async def get_features(self, symbol: str, feature_names: List[str]) -> Dict[str, float]:
        """获取特征"""
        pass

    @abstractmethod
    async def get_batch_features(self, symbols: List[str], feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """批量获取特征"""
        pass


class ModelProvider(ABC):
    """模型提供者接口"""

    @abstractmethod
    async def load_model(self, model_id: str, version: str) -> Any:
        """加载模型"""
        pass

    @abstractmethod
    async def predict(self, model: Any, features: Dict[str, float]) -> Tuple[Any, float]:
        """执行预测，返回(预测值, 置信度)"""
        pass

    @abstractmethod
    async def predict_batch(self, model: Any, features_list: List[Dict[str, float]]) -> List[Tuple[Any, float]]:
        """批量预测"""
        pass


class SimulatedFeatureProvider(FeatureProvider):
    """模拟特征提供者"""

    def __init__(self):
        self._features: Dict[str, Dict[str, float]] = {}

    def set_features(self, symbol: str, features: Dict[str, float]) -> None:
        self._features[symbol] = features

    async def get_features(self, symbol: str, feature_names: List[str]) -> Dict[str, float]:
        all_features = self._features.get(symbol, {})
        return {name: all_features.get(name, 0.0) for name in feature_names}

    async def get_batch_features(self, symbols: List[str], feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_features(symbol, feature_names)
        return result


class SimulatedModelProvider(ModelProvider):
    """模拟模型提供者"""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._predict_fn: Optional[Callable] = None

    def set_model(self, model_id: str, version: str, model: Any) -> None:
        key = f"{model_id}:{version}"
        self._models[key] = model

    def set_predict_function(self, fn: Callable) -> None:
        """设置自定义预测函数"""
        self._predict_fn = fn

    async def load_model(self, model_id: str, version: str) -> Any:
        key = f"{model_id}:{version}"
        return self._models.get(key)

    async def predict(self, model: Any, features: Dict[str, float]) -> Tuple[Any, float]:
        if self._predict_fn:
            return self._predict_fn(model, features)
        # 默认模拟：基于特征均值生成预测
        if features:
            avg = statistics.mean(features.values())
            # 归一化到 -1 到 1
            value = max(-1, min(1, avg / 100))
            confidence = 0.5 + abs(value) * 0.4
            return value, confidence
        return 0.0, 0.5

    async def predict_batch(self, model: Any, features_list: List[Dict[str, float]]) -> List[Tuple[Any, float]]:
        results = []
        for features in features_list:
            results.append(await self.predict(model, features))
        return results


# ============ Main Class ============

class MLInferenceService:
    """机器学习推理服务"""

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        feature_provider: Optional[FeatureProvider] = None,
        model_provider: Optional[ModelProvider] = None,
        signal_callback: Optional[Callable[[TradingSignal], None]] = None,
        drift_callback: Optional[Callable[[DriftAlert], None]] = None,
    ):
        self.config = config or InferenceConfig()
        self.feature_provider = feature_provider or simulated_default(SimulatedFeatureProvider, "data from MLInferenceService's simulated feature_provider (SimulatedFeatureProvider)")
        self.model_provider = model_provider or simulated_default(SimulatedModelProvider, "data from MLInferenceService's simulated model_provider (SimulatedModelProvider)")
        self.signal_callback = signal_callback
        self.drift_callback = drift_callback

        # 状态
        self._state = ServiceState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._models: Dict[str, ModelMetadata] = {}
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._loaded_models: Dict[str, Any] = {}  # model_id:version -> model
        self._cache: Dict[str, Tuple[Prediction, datetime]] = {}  # cache_key -> (prediction, timestamp)
        self._predictions: deque = deque(maxlen=self.config.history_size)
        self._signals: deque = deque(maxlen=self.config.history_size)
        self._alerts: deque = deque(maxlen=1000)
        self._latencies: Dict[str, deque] = {}  # model_id -> latencies
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}  # model_id -> metrics

    @property
    def state(self) -> ServiceState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动服务"""
        if self._state == ServiceState.RUNNING:
            return

        self._state = ServiceState.STARTING
        logger.info("Starting ML Inference Service")

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._state = ServiceState.RUNNING
        logger.info("ML Inference Service started")

    async def stop(self) -> None:
        """停止服务"""
        if self._state == ServiceState.STOPPED:
            return

        logger.info("Stopping ML Inference Service")
        self._state = ServiceState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("ML Inference Service stopped")

    async def pause(self) -> None:
        """暂停服务"""
        if self._state == ServiceState.RUNNING:
            self._state = ServiceState.PAUSED
            logger.info("ML Inference Service paused")

    async def resume(self) -> None:
        """恢复服务"""
        if self._state == ServiceState.PAUSED:
            self._state = ServiceState.RUNNING
            logger.info("ML Inference Service resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (ServiceState.RUNNING, ServiceState.PAUSED):
            try:
                if self._state == ServiceState.RUNNING:
                    await self._check_model_drift()
                    await self._cleanup_cache()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

    # ============ Model Registry ============

    def register_model(self, metadata: ModelMetadata) -> None:
        """注册模型"""
        self._models[metadata.model_id] = metadata
        self._versions[metadata.model_id] = []
        self._latencies[metadata.model_id] = deque(maxlen=1000)
        logger.info(f"Registered model: {metadata.model_id}")

    def add_model_version(self, version: ModelVersion) -> None:
        """添加模型版本"""
        if version.model_id not in self._models:
            raise ValueError(f"Model not registered: {version.model_id}")

        versions = self._versions.get(version.model_id, [])
        versions.append(version)
        self._versions[version.model_id] = versions
        logger.info(f"Added version {version.version} for model {version.model_id}")

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self._models.get(model_id)

    def get_all_models(self) -> List[ModelMetadata]:
        """获取所有模型"""
        return list(self._models.values())

    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """获取模型所有版本"""
        return self._versions.get(model_id, [])

    def get_champion_version(self, model_id: str) -> Optional[ModelVersion]:
        """获取主力版本"""
        versions = self._versions.get(model_id, [])
        for v in versions:
            if v.is_champion:
                return v
        # 如果没有标记为champion，返回最新版本
        if versions:
            return versions[-1]
        return None

    def set_champion_version(self, model_id: str, version: str) -> bool:
        """设置主力版本"""
        versions = self._versions.get(model_id, [])
        found = False
        for v in versions:
            if v.version == version:
                v.is_champion = True
                found = True
            else:
                v.is_champion = False
        return found

    async def load_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """加载模型"""
        if version is None:
            champion = self.get_champion_version(model_id)
            if champion is None:
                return False
            version = champion.version

        key = f"{model_id}:{version}"
        if key in self._loaded_models:
            return True

        try:
            model = await self.model_provider.load_model(model_id, version)
            if model is not None:
                self._loaded_models[key] = model

                # 更新版本状态
                for v in self._versions.get(model_id, []):
                    if v.version == version:
                        v.status = ModelStatus.SERVING
                        v.deployed_at = datetime.now()

                logger.info(f"Loaded model {model_id} version {version}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            return False

    def unload_model(self, model_id: str, version: str) -> bool:
        """卸载模型"""
        key = f"{model_id}:{version}"
        if key in self._loaded_models:
            del self._loaded_models[key]

            # 更新版本状态
            for v in self._versions.get(model_id, []):
                if v.version == version:
                    v.status = ModelStatus.REGISTERED

            logger.info(f"Unloaded model {model_id} version {version}")
            return True
        return False

    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self._loaded_models.keys())

    # ============ Inference ============

    async def predict(
        self,
        model_id: str,
        symbol: str,
        features: Optional[Dict[str, float]] = None,
        version: Optional[str] = None,
    ) -> Optional[Prediction]:
        """执行预测"""
        if self._state != ServiceState.RUNNING:
            return None

        # 获取模型和版本
        if version is None:
            champion = self.get_champion_version(model_id)
            if champion is None:
                return None
            version = champion.version

        key = f"{model_id}:{version}"
        model = self._loaded_models.get(key)
        if model is None:
            # 尝试加载
            if not await self.load_model(model_id, version):
                return None
            model = self._loaded_models.get(key)

        # 获取特征
        if features is None:
            metadata = self._models.get(model_id)
            if metadata and metadata.features:
                features = await self.feature_provider.get_features(symbol, metadata.features)
            else:
                features = {}

        feature_vector = FeatureVector(symbol=symbol, features=features)

        # 检查缓存
        if self.config.enable_caching:
            cache_key = f"{key}:{feature_vector.feature_hash}"
            cached = self._cache.get(cache_key)
            if cached:
                prediction, cache_time = cached
                if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                    return prediction

        # 执行推理
        start_time = datetime.now()
        try:
            value, confidence = await self.model_provider.predict(model, features)
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # 记录延迟
            if model_id in self._latencies:
                self._latencies[model_id].append(latency_ms)

            # 确定预测类型
            metadata = self._models.get(model_id)
            if metadata:
                if metadata.model_type == ModelType.CLASSIFICATION:
                    pred_type = PredictionType.DIRECTION
                elif metadata.model_type == ModelType.TIME_SERIES:
                    pred_type = PredictionType.PRICE_TARGET
                else:
                    pred_type = PredictionType.SCORE
            else:
                pred_type = PredictionType.SCORE

            prediction = Prediction(
                prediction_id=str(uuid.uuid4())[:8],
                model_id=model_id,
                model_version=version,
                symbol=symbol,
                prediction_type=pred_type,
                value=value,
                confidence=confidence,
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                feature_hash=feature_vector.feature_hash,
            )

            # 更新缓存
            if self.config.enable_caching:
                cache_key = f"{key}:{feature_vector.feature_hash}"
                self._cache[cache_key] = (prediction, datetime.now())

            # 保存历史
            self._predictions.append(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Prediction error for {model_id}: {e}")
            return None

    async def predict_batch(
        self,
        model_id: str,
        symbols: List[str],
        version: Optional[str] = None,
    ) -> Dict[str, Prediction]:
        """批量预测"""
        results = {}
        for symbol in symbols:
            prediction = await self.predict(model_id, symbol, version=version)
            if prediction:
                results[symbol] = prediction
        return results

    # ============ Signal Generation ============

    async def generate_signal(
        self,
        model_id: str,
        symbol: str,
        current_price: Optional[float] = None,
        version: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """生成交易信号"""
        prediction = await self.predict(model_id, symbol, version=version)
        if prediction is None:
            return None

        # 转换预测为信号
        signal = self._prediction_to_signal(prediction, current_price)
        if signal:
            self._signals.append(signal)

            # 回调
            if self.signal_callback:
                self.signal_callback(signal)

        return signal

    def _prediction_to_signal(
        self,
        prediction: Prediction,
        current_price: Optional[float] = None,
    ) -> Optional[TradingSignal]:
        """将预测转换为交易信号"""
        # 确定方向和强度
        value = prediction.value
        if isinstance(value, (int, float)):
            if value > 0.5:
                direction = SignalDirection.STRONG_LONG
                strength = min(1.0, value)
            elif value > 0.2:
                direction = SignalDirection.LONG
                strength = value
            elif value < -0.5:
                direction = SignalDirection.STRONG_SHORT
                strength = min(1.0, abs(value))
            elif value < -0.2:
                direction = SignalDirection.SHORT
                strength = abs(value)
            else:
                direction = SignalDirection.NEUTRAL
                strength = 0.0
        else:
            # 非数值预测
            direction = SignalDirection.NEUTRAL
            strength = 0.0

        # 计算目标价和止损
        target_price = None
        stop_loss = None
        take_profit = None

        if current_price and strength > 0:
            if direction in (SignalDirection.STRONG_LONG, SignalDirection.LONG):
                target_price = current_price * (1 + 0.02 * strength)  # 2% * strength
                stop_loss = current_price * (1 - 0.01 * strength)     # 1% * strength
                take_profit = current_price * (1 + 0.03 * strength)   # 3% * strength
            elif direction in (SignalDirection.STRONG_SHORT, SignalDirection.SHORT):
                target_price = current_price * (1 - 0.02 * strength)
                stop_loss = current_price * (1 + 0.01 * strength)
                take_profit = current_price * (1 - 0.03 * strength)

        # 计算过期时间
        expires_at = datetime.now() + timedelta(days=1)

        signal = TradingSignal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=prediction.symbol,
            direction=direction,
            strength=strength,
            confidence=prediction.confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_horizon="1d",
            source_model=prediction.model_id,
            source_prediction_id=prediction.prediction_id,
            expires_at=expires_at,
        )

        return signal

    def get_signals(
        self,
        symbol: Optional[str] = None,
        model_id: Optional[str] = None,
        direction: Optional[str] = None,
        min_strength: float = 0.0,
        min_confidence: float = 0.0,
        include_expired: bool = False,
        limit: int = 100,
    ) -> List[TradingSignal]:
        """获取交易信号"""
        signals = list(self._signals)

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if model_id:
            signals = [s for s in signals if s.source_model == model_id]
        if direction:
            signals = [s for s in signals if s.direction.value == direction]
        if min_strength > 0:
            signals = [s for s in signals if s.strength >= min_strength]
        if min_confidence > 0:
            signals = [s for s in signals if s.confidence >= min_confidence]
        if not include_expired:
            signals = [s for s in signals if not s.is_expired]

        return signals[-limit:]

    # ============ Drift Detection ============

    def set_baseline_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """设置基线指标"""
        self._baseline_metrics[model_id] = metrics

    async def _check_model_drift(self) -> None:
        """检查模型漂移"""
        if not self.config.enable_drift_detection:
            return

        for model_id, baseline in self._baseline_metrics.items():
            current_metrics = self._calculate_current_metrics(model_id)

            for metric_name, baseline_value in baseline.items():
                current_value = current_metrics.get(metric_name, baseline_value)

                if baseline_value != 0:
                    drift = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    drift = abs(current_value)

                if drift > self.config.drift_threshold:
                    self._create_drift_alert(
                        model_id=model_id,
                        metric_name=metric_name,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        drift=drift,
                    )

    def _calculate_current_metrics(self, model_id: str) -> Dict[str, float]:
        """计算当前指标"""
        recent_predictions = [
            p for p in self._predictions
            if p.model_id == model_id
        ][-100:]  # 最近100个预测

        if not recent_predictions:
            return {}

        # 计算各种指标
        values = [p.value for p in recent_predictions if isinstance(p.value, (int, float))]
        confidences = [p.confidence for p in recent_predictions]
        latencies = [p.latency_ms for p in recent_predictions]

        metrics = {}
        if values:
            metrics["mean_prediction"] = statistics.mean(values)
            metrics["std_prediction"] = statistics.stdev(values) if len(values) > 1 else 0
        if confidences:
            metrics["mean_confidence"] = statistics.mean(confidences)
        if latencies:
            metrics["mean_latency"] = statistics.mean(latencies)

        return metrics

    def _create_drift_alert(
        self,
        model_id: str,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        drift: float,
    ) -> None:
        """创建漂移告警"""
        # 确定严重程度
        if drift > self.config.drift_threshold * 2:
            severity = AlertSeverity.CRITICAL
        elif drift > self.config.drift_threshold * 1.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        # 确定漂移类型
        if "prediction" in metric_name.lower():
            drift_type = DriftType.PREDICTION_DRIFT
        elif "latency" in metric_name.lower() or "performance" in metric_name.lower():
            drift_type = DriftType.PERFORMANCE_DRIFT
        else:
            drift_type = DriftType.FEATURE_DRIFT

        alert = DriftAlert(
            alert_id=str(uuid.uuid4())[:8],
            model_id=model_id,
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold=self.config.drift_threshold,
            description=f"Model {model_id} drift detected: {metric_name} changed from {baseline_value:.4f} to {current_value:.4f} ({drift:.1%})",
        )

        self._alerts.append(alert)

        if self.drift_callback:
            self.drift_callback(alert)

    def get_drift_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[DriftAlert]:
        """获取漂移告警"""
        alerts = list(self._alerts)

        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                return True
        return False

    # ============ Cache Management ============

    async def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = []

        for key, (_, cache_time) in self._cache.items():
            if (now - cache_time).total_seconds() > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        # 限制缓存大小
        if len(self._cache) > self.config.max_cache_size:
            # 删除最旧的条目
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1],  # 按时间排序
            )
            for key, _ in sorted_items[:len(self._cache) - self.config.max_cache_size]:
                del self._cache[key]

    def clear_cache(self, model_id: Optional[str] = None) -> int:
        """清除缓存"""
        if model_id:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(model_id)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)
        else:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": self.config.max_cache_size,
            "ttl_seconds": self.config.cache_ttl_seconds,
        }

    # ============ Performance Metrics ============

    def get_model_performance(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """获取模型性能指标"""
        if model_id not in self._models:
            return None

        # 获取版本
        champion = self.get_champion_version(model_id)
        version = champion.version if champion else "unknown"

        # 获取最近的预测
        predictions = [p for p in self._predictions if p.model_id == model_id]
        latencies = list(self._latencies.get(model_id, []))

        if not predictions:
            return ModelPerformanceMetrics(
                model_id=model_id,
                model_version=version,
            )

        # 计算延迟指标
        if latencies:
            sorted_latencies = sorted(latencies)
            avg_latency = statistics.mean(latencies)
            p50_idx = int(len(sorted_latencies) * 0.5)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p50_latency = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
            p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0

        # 计算缓存命中率
        cache_hits = sum(1 for k in self._cache.keys() if k.startswith(model_id))
        cache_hit_rate = cache_hits / len(predictions) if predictions else 0

        return ModelPerformanceMetrics(
            model_id=model_id,
            model_version=version,
            total_predictions=len(predictions),
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            cache_hit_rate=cache_hit_rate,
            last_prediction_at=predictions[-1].timestamp if predictions else None,
        )

    # ============ Statistics ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "registered_models": len(self._models),
            "loaded_models": len(self._loaded_models),
            "total_predictions": len(self._predictions),
            "total_signals": len(self._signals),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "cache_size": len(self._cache),
            "config": self.config.to_dict(),
        }

    def get_predictions_history(
        self,
        model_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Prediction]:
        """获取预测历史"""
        predictions = list(self._predictions)

        if model_id:
            predictions = [p for p in predictions if p.model_id == model_id]
        if symbol:
            predictions = [p for p in predictions if p.symbol == symbol]

        return predictions[-limit:]


# ============ Factory Functions ============

def create_inference_service(
    feature_provider: Optional[FeatureProvider] = None,
    model_provider: Optional[ModelProvider] = None,
    **config_kwargs,
) -> MLInferenceService:
    """创建ML推理服务"""
    config = InferenceConfig(**config_kwargs)
    return MLInferenceService(
        config=config,
        feature_provider=feature_provider,
        model_provider=model_provider,
    )


def create_model_metadata(
    model_id: str,
    name: str,
    model_type: str,
    version: str,
    **kwargs,
) -> ModelMetadata:
    """创建模型元数据"""
    return ModelMetadata(
        model_id=model_id,
        name=name,
        model_type=ModelType(model_type),
        version=version,
        **kwargs,
    )


def create_model_version(
    model_id: str,
    version: str,
    **kwargs,
) -> ModelVersion:
    """创建模型版本"""
    return ModelVersion(
        version_id=str(uuid.uuid4())[:8],
        model_id=model_id,
        version=version,
        status=ModelStatus.REGISTERED,
        **kwargs,
    )


async def quick_predict(
    service: MLInferenceService,
    model_id: str,
    symbol: str,
    features: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """快速预测"""
    prediction = await service.predict(model_id, symbol, features)
    if prediction:
        return prediction.to_dict()
    return None


# ============ Exports ============

__all__ = [
    # Enums
    "ModelType",
    "ModelStatus",
    "PredictionType",
    "SignalDirection",
    "DriftType",
    "AlertSeverity",
    "ServiceState",
    # Data Classes
    "ModelMetadata",
    "ModelVersion",
    "FeatureVector",
    "Prediction",
    "TradingSignal",
    "DriftAlert",
    "ModelPerformanceMetrics",
    "InferenceConfig",
    # Providers
    "FeatureProvider",
    "ModelProvider",
    "SimulatedFeatureProvider",
    "SimulatedModelProvider",
    # Main Class
    "MLInferenceService",
    # Factory Functions
    "create_inference_service",
    "create_model_metadata",
    "create_model_version",
    "quick_predict",
]
