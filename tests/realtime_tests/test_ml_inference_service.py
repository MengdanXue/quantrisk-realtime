"""
ML Inference Service Tests - 机器学习推理服务测试

测试覆盖：
- 服务生命周期管理
- 模型注册与版本管理
- 特征与模型提供者
- 预测与信号生成
- 缓存管理
- 漂移检测
- 性能指标
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from quantrisk_realtime.realtime.ml_inference_service import (
    # Enums
    ModelType,
    ModelStatus,
    PredictionType,
    SignalDirection,
    DriftType,
    AlertSeverity,
    ServiceState,
    # Data Classes
    ModelMetadata,
    ModelVersion,
    FeatureVector,
    Prediction,
    TradingSignal,
    DriftAlert,
    ModelPerformanceMetrics,
    InferenceConfig,
    # Providers
    FeatureProvider,
    ModelProvider,
    SimulatedFeatureProvider,
    SimulatedModelProvider,
    # Main Class
    MLInferenceService,
    # Factory Functions
    create_inference_service,
    create_model_metadata,
    create_model_version,
    quick_predict,
)


# ============ Fixtures ============


@pytest.fixture
def sample_config():
    """创建测试配置"""
    return InferenceConfig(
        cache_ttl_seconds=60,
        max_cache_size=1000,
        batch_size=32,
        timeout_ms=1000,
        enable_caching=True,
        enable_drift_detection=True,
        drift_threshold=0.1,
        confidence_threshold=0.5,
        history_size=1000,
        update_interval=0.1,
    )


@pytest.fixture
def sample_metadata():
    """创建测试模型元数据"""
    return ModelMetadata(
        model_id="MODEL001",
        name="Test Signal Model",
        model_type=ModelType.CLASSIFICATION,
        version="1.0.0",
        description="Test classification model",
        features=["momentum", "volatility", "trend"],
        target="direction",
        framework="sklearn",
        metrics={"accuracy": 0.85, "f1": 0.82},
        tags=["production", "signal"],
    )


@pytest.fixture
def sample_version():
    """创建测试模型版本"""
    return ModelVersion(
        version_id="VER001",
        model_id="MODEL001",
        version="1.0.0",
        status=ModelStatus.REGISTERED,
        artifact_path="/models/model001/v1",
        metrics={"accuracy": 0.85},
        is_champion=True,
    )


@pytest.fixture
def feature_provider():
    """创建特征提供者"""
    provider = SimulatedFeatureProvider()
    provider.set_features("AAPL", {
        "momentum": 0.5,
        "volatility": 0.2,
        "trend": 0.8,
    })
    provider.set_features("GOOGL", {
        "momentum": -0.3,
        "volatility": 0.4,
        "trend": -0.2,
    })
    return provider


@pytest.fixture
def model_provider():
    """创建模型提供者"""
    provider = SimulatedModelProvider()
    provider.set_model("MODEL001", "1.0.0", {"type": "test_model"})
    return provider


@pytest.fixture
def service(sample_config, feature_provider, model_provider):
    """创建推理服务"""
    return MLInferenceService(
        config=sample_config,
        feature_provider=feature_provider,
        model_provider=model_provider,
    )


@pytest.fixture
def service_with_model(service, sample_metadata, sample_version):
    """创建带模型的推理服务"""
    service.register_model(sample_metadata)
    service.add_model_version(sample_version)
    return service


# ============ Enum Tests ============


class TestEnums:
    """枚举类测试"""

    def test_model_type_values(self):
        """测试模型类型枚举值"""
        assert ModelType.CLASSIFICATION.value == "classification"
        assert ModelType.REGRESSION.value == "regression"
        assert ModelType.RANKING.value == "ranking"
        assert ModelType.TIME_SERIES.value == "time_series"
        assert ModelType.REINFORCEMENT.value == "reinforcement"
        assert ModelType.ENSEMBLE.value == "ensemble"

    def test_model_status_values(self):
        """测试模型状态枚举值"""
        assert ModelStatus.REGISTERED.value == "registered"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.SERVING.value == "serving"
        assert ModelStatus.PAUSED.value == "paused"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.ERROR.value == "error"

    def test_prediction_type_values(self):
        """测试预测类型枚举值"""
        assert PredictionType.SIGNAL.value == "signal"
        assert PredictionType.DIRECTION.value == "direction"
        assert PredictionType.PRICE_TARGET.value == "price_target"
        assert PredictionType.VOLATILITY.value == "volatility"
        assert PredictionType.PROBABILITY.value == "probability"
        assert PredictionType.SCORE.value == "score"
        assert PredictionType.RANKING.value == "ranking"

    def test_signal_direction_values(self):
        """测试信号方向枚举值"""
        assert SignalDirection.STRONG_LONG.value == "strong_long"
        assert SignalDirection.LONG.value == "long"
        assert SignalDirection.NEUTRAL.value == "neutral"
        assert SignalDirection.SHORT.value == "short"
        assert SignalDirection.STRONG_SHORT.value == "strong_short"

    def test_drift_type_values(self):
        """测试漂移类型枚举值"""
        assert DriftType.FEATURE_DRIFT.value == "feature_drift"
        assert DriftType.PREDICTION_DRIFT.value == "prediction_drift"
        assert DriftType.PERFORMANCE_DRIFT.value == "performance_drift"
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"

    def test_alert_severity_values(self):
        """测试告警严重程度枚举值"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_service_state_values(self):
        """测试服务状态枚举值"""
        assert ServiceState.STOPPED.value == "stopped"
        assert ServiceState.STARTING.value == "starting"
        assert ServiceState.RUNNING.value == "running"
        assert ServiceState.PAUSED.value == "paused"
        assert ServiceState.ERROR.value == "error"


# ============ Data Class Tests ============


class TestModelMetadata:
    """模型元数据测试"""

    def test_creation(self, sample_metadata):
        """测试创建"""
        assert sample_metadata.model_id == "MODEL001"
        assert sample_metadata.name == "Test Signal Model"
        assert sample_metadata.model_type == ModelType.CLASSIFICATION
        assert sample_metadata.version == "1.0.0"

    def test_to_dict(self, sample_metadata):
        """测试字典转换"""
        data = sample_metadata.to_dict()
        assert data["model_id"] == "MODEL001"
        assert data["model_type"] == "classification"
        assert "created_at" in data

    def test_default_values(self):
        """测试默认值"""
        metadata = ModelMetadata(
            model_id="M1",
            name="Test",
            model_type=ModelType.REGRESSION,
            version="1.0",
        )
        assert metadata.description == ""
        assert metadata.features == []
        assert metadata.framework == "sklearn"


class TestModelVersion:
    """模型版本测试"""

    def test_creation(self, sample_version):
        """测试创建"""
        assert sample_version.version_id == "VER001"
        assert sample_version.model_id == "MODEL001"
        assert sample_version.version == "1.0.0"
        assert sample_version.status == ModelStatus.REGISTERED

    def test_to_dict(self, sample_version):
        """测试字典转换"""
        data = sample_version.to_dict()
        assert data["version_id"] == "VER001"
        assert data["status"] == "registered"
        assert data["is_champion"] is True


class TestFeatureVector:
    """特征向量测试"""

    def test_creation(self):
        """测试创建"""
        fv = FeatureVector(
            symbol="AAPL",
            features={"momentum": 0.5, "volatility": 0.2},
        )
        assert fv.symbol == "AAPL"
        assert fv.features["momentum"] == 0.5
        assert fv.feature_hash != ""

    def test_hash_computation(self):
        """测试哈希计算"""
        fv1 = FeatureVector(symbol="AAPL", features={"a": 1.0})
        fv2 = FeatureVector(symbol="AAPL", features={"a": 1.0})
        fv3 = FeatureVector(symbol="AAPL", features={"a": 2.0})
        assert fv1.feature_hash == fv2.feature_hash
        assert fv1.feature_hash != fv3.feature_hash

    def test_to_dict(self):
        """测试字典转换"""
        fv = FeatureVector(symbol="AAPL", features={"x": 1.0})
        data = fv.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["feature_hash"] != ""


class TestPrediction:
    """预测结果测试"""

    def test_creation(self):
        """测试创建"""
        pred = Prediction(
            prediction_id="PRED001",
            model_id="MODEL001",
            model_version="1.0.0",
            symbol="AAPL",
            prediction_type=PredictionType.DIRECTION,
            value=0.8,
            confidence=0.9,
        )
        assert pred.prediction_id == "PRED001"
        assert pred.value == 0.8
        assert pred.confidence == 0.9

    def test_to_dict(self):
        """测试字典转换"""
        pred = Prediction(
            prediction_id="PRED001",
            model_id="MODEL001",
            model_version="1.0.0",
            symbol="AAPL",
            prediction_type=PredictionType.SCORE,
            value=0.5,
            confidence=0.7,
        )
        data = pred.to_dict()
        assert data["prediction_type"] == "score"
        assert "timestamp" in data


class TestTradingSignal:
    """交易信号测试"""

    def test_creation(self):
        """测试创建"""
        signal = TradingSignal(
            signal_id="SIG001",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.7,
            confidence=0.85,
            target_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
        )
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.strength == 0.7

    def test_is_expired(self):
        """测试过期检查"""
        # 未过期信号
        signal1 = TradingSignal(
            signal_id="SIG001",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.8,
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert signal1.is_expired is False

        # 已过期信号
        signal2 = TradingSignal(
            signal_id="SIG002",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            strength=0.5,
            confidence=0.8,
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert signal2.is_expired is True

        # 无过期时间
        signal3 = TradingSignal(
            signal_id="SIG003",
            symbol="AAPL",
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            confidence=0.5,
        )
        assert signal3.is_expired is False

    def test_to_dict(self):
        """测试字典转换"""
        signal = TradingSignal(
            signal_id="SIG001",
            symbol="AAPL",
            direction=SignalDirection.STRONG_LONG,
            strength=0.9,
            confidence=0.95,
        )
        data = signal.to_dict()
        assert data["direction"] == "strong_long"
        assert "is_expired" in data


class TestDriftAlert:
    """漂移告警测试"""

    def test_creation(self):
        """测试创建"""
        alert = DriftAlert(
            alert_id="ALERT001",
            model_id="MODEL001",
            drift_type=DriftType.PREDICTION_DRIFT,
            severity=AlertSeverity.WARNING,
            metric_name="mean_prediction",
            baseline_value=0.5,
            current_value=0.7,
            threshold=0.1,
            description="Drift detected",
        )
        assert alert.alert_id == "ALERT001"
        assert alert.drift_type == DriftType.PREDICTION_DRIFT

    def test_drift_magnitude(self):
        """测试漂移幅度计算"""
        alert = DriftAlert(
            alert_id="ALERT001",
            model_id="MODEL001",
            drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.INFO,
            metric_name="test",
            baseline_value=100.0,
            current_value=120.0,
            threshold=0.1,
            description="Test",
        )
        assert alert.drift_magnitude == 0.2  # |120-100|/100 = 0.2

    def test_drift_magnitude_zero_baseline(self):
        """测试零基线的漂移幅度"""
        alert = DriftAlert(
            alert_id="ALERT001",
            model_id="MODEL001",
            drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.INFO,
            metric_name="test",
            baseline_value=0.0,
            current_value=0.5,
            threshold=0.1,
            description="Test",
        )
        assert alert.drift_magnitude == 0.5


class TestModelPerformanceMetrics:
    """模型性能指标测试"""

    def test_creation(self):
        """测试创建"""
        metrics = ModelPerformanceMetrics(
            model_id="MODEL001",
            model_version="1.0.0",
            total_predictions=1000,
            avg_latency_ms=5.5,
            p95_latency_ms=15.0,
            cache_hit_rate=0.75,
        )
        assert metrics.total_predictions == 1000
        assert metrics.avg_latency_ms == 5.5

    def test_to_dict(self):
        """测试字典转换"""
        metrics = ModelPerformanceMetrics(
            model_id="MODEL001",
            model_version="1.0.0",
        )
        data = metrics.to_dict()
        assert data["model_id"] == "MODEL001"
        assert "calculated_at" in data


class TestInferenceConfig:
    """推理配置测试"""

    def test_creation(self, sample_config):
        """测试创建"""
        assert sample_config.cache_ttl_seconds == 60
        assert sample_config.enable_caching is True

    def test_default_values(self):
        """测试默认值"""
        config = InferenceConfig()
        assert config.cache_ttl_seconds == 60
        assert config.max_cache_size == 10000
        assert config.batch_size == 32

    def test_to_dict(self, sample_config):
        """测试字典转换"""
        data = sample_config.to_dict()
        assert data["cache_ttl_seconds"] == 60
        assert data["enable_drift_detection"] is True


# ============ Provider Tests ============


class TestSimulatedFeatureProvider:
    """模拟特征提供者测试"""

    @pytest.mark.asyncio
    async def test_set_and_get_features(self, feature_provider):
        """测试设置和获取特征"""
        features = await feature_provider.get_features("AAPL", ["momentum", "volatility"])
        assert features["momentum"] == 0.5
        assert features["volatility"] == 0.2

    @pytest.mark.asyncio
    async def test_get_missing_features(self, feature_provider):
        """测试获取不存在的特征"""
        features = await feature_provider.get_features("AAPL", ["unknown_feature"])
        assert features["unknown_feature"] == 0.0

    @pytest.mark.asyncio
    async def test_get_features_unknown_symbol(self, feature_provider):
        """测试获取未知标的特征"""
        features = await feature_provider.get_features("UNKNOWN", ["momentum"])
        assert features["momentum"] == 0.0

    @pytest.mark.asyncio
    async def test_batch_features(self, feature_provider):
        """测试批量获取特征"""
        features = await feature_provider.get_batch_features(
            ["AAPL", "GOOGL"],
            ["momentum", "volatility"],
        )
        assert "AAPL" in features
        assert "GOOGL" in features
        assert features["AAPL"]["momentum"] == 0.5
        assert features["GOOGL"]["momentum"] == -0.3


class TestSimulatedModelProvider:
    """模拟模型提供者测试"""

    @pytest.mark.asyncio
    async def test_set_and_load_model(self, model_provider):
        """测试设置和加载模型"""
        model = await model_provider.load_model("MODEL001", "1.0.0")
        assert model == {"type": "test_model"}

    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, model_provider):
        """测试加载不存在的模型"""
        model = await model_provider.load_model("UNKNOWN", "1.0.0")
        assert model is None

    @pytest.mark.asyncio
    async def test_default_predict(self, model_provider):
        """测试默认预测"""
        model = await model_provider.load_model("MODEL001", "1.0.0")
        value, confidence = await model_provider.predict(model, {"x": 50.0})
        assert -1 <= value <= 1
        assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    async def test_custom_predict_function(self, model_provider):
        """测试自定义预测函数"""
        def custom_fn(model, features):
            return 0.99, 0.95

        model_provider.set_predict_function(custom_fn)
        model = await model_provider.load_model("MODEL001", "1.0.0")
        value, confidence = await model_provider.predict(model, {"x": 1.0})
        assert value == 0.99
        assert confidence == 0.95

    @pytest.mark.asyncio
    async def test_batch_predict(self, model_provider):
        """测试批量预测"""
        model = await model_provider.load_model("MODEL001", "1.0.0")
        results = await model_provider.predict_batch(
            model,
            [{"x": 50.0}, {"x": 100.0}],
        )
        assert len(results) == 2


# ============ Service Lifecycle Tests ============


class TestServiceLifecycle:
    """服务生命周期测试"""

    def test_initial_state(self, service):
        """测试初始状态"""
        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_start_stop(self, service):
        """测试启动和停止"""
        await service.start()
        assert service.state == ServiceState.RUNNING

        await service.stop()
        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_resume(self, service):
        """测试暂停和恢复"""
        await service.start()

        await service.pause()
        assert service.state == ServiceState.PAUSED

        await service.resume()
        assert service.state == ServiceState.RUNNING

        await service.stop()

    @pytest.mark.asyncio
    async def test_double_start(self, service):
        """测试重复启动"""
        await service.start()
        await service.start()  # Should be no-op
        assert service.state == ServiceState.RUNNING
        await service.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self, service):
        """测试重复停止"""
        await service.start()
        await service.stop()
        await service.stop()  # Should be no-op
        assert service.state == ServiceState.STOPPED


# ============ Model Registry Tests ============


class TestModelRegistry:
    """模型注册测试"""

    def test_register_model(self, service, sample_metadata):
        """测试注册模型"""
        service.register_model(sample_metadata)
        assert service.get_model("MODEL001") == sample_metadata

    def test_get_all_models(self, service, sample_metadata):
        """测试获取所有模型"""
        service.register_model(sample_metadata)
        models = service.get_all_models()
        assert len(models) == 1
        assert models[0].model_id == "MODEL001"

    def test_add_model_version(self, service, sample_metadata, sample_version):
        """测试添加模型版本"""
        service.register_model(sample_metadata)
        service.add_model_version(sample_version)
        versions = service.get_model_versions("MODEL001")
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"

    def test_add_version_unregistered_model(self, service, sample_version):
        """测试为未注册模型添加版本"""
        with pytest.raises(ValueError):
            service.add_model_version(sample_version)

    def test_get_champion_version(self, service_with_model):
        """测试获取主力版本"""
        champion = service_with_model.get_champion_version("MODEL001")
        assert champion is not None
        assert champion.version == "1.0.0"
        assert champion.is_champion is True

    def test_set_champion_version(self, service_with_model):
        """测试设置主力版本"""
        # 添加新版本
        new_version = ModelVersion(
            version_id="VER002",
            model_id="MODEL001",
            version="2.0.0",
            status=ModelStatus.REGISTERED,
        )
        service_with_model.add_model_version(new_version)

        # 设置新版本为主力
        result = service_with_model.set_champion_version("MODEL001", "2.0.0")
        assert result is True

        champion = service_with_model.get_champion_version("MODEL001")
        assert champion.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_load_model(self, service_with_model):
        """测试加载模型"""
        result = await service_with_model.load_model("MODEL001", "1.0.0")
        assert result is True
        assert "MODEL001:1.0.0" in service_with_model.get_loaded_models()

    @pytest.mark.asyncio
    async def test_load_champion_version(self, service_with_model):
        """测试加载主力版本"""
        result = await service_with_model.load_model("MODEL001")
        assert result is True

    def test_unload_model(self, service_with_model):
        """测试卸载模型"""
        # 先手动添加到已加载列表
        service_with_model._loaded_models["MODEL001:1.0.0"] = {"test": True}

        result = service_with_model.unload_model("MODEL001", "1.0.0")
        assert result is True
        assert "MODEL001:1.0.0" not in service_with_model.get_loaded_models()


# ============ Prediction Tests ============


class TestPredictions:
    """预测测试"""

    @pytest.mark.asyncio
    async def test_predict(self, service_with_model):
        """测试预测"""
        await service_with_model.start()

        prediction = await service_with_model.predict(
            model_id="MODEL001",
            symbol="AAPL",
            features={"momentum": 0.5, "volatility": 0.2},
        )

        assert prediction is not None
        assert prediction.model_id == "MODEL001"
        assert prediction.symbol == "AAPL"

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_predict_not_running(self, service_with_model):
        """测试服务未运行时预测"""
        prediction = await service_with_model.predict(
            model_id="MODEL001",
            symbol="AAPL",
        )
        assert prediction is None

    @pytest.mark.asyncio
    async def test_predict_caching(self, service_with_model):
        """测试预测缓存"""
        await service_with_model.start()

        features = {"momentum": 0.5, "volatility": 0.2}

        # 首次预测
        pred1 = await service_with_model.predict("MODEL001", "AAPL", features)
        # 再次预测（应该命中缓存）
        pred2 = await service_with_model.predict("MODEL001", "AAPL", features)

        assert pred1.feature_hash == pred2.feature_hash

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_predict_batch(self, service_with_model):
        """测试批量预测"""
        await service_with_model.start()

        results = await service_with_model.predict_batch(
            model_id="MODEL001",
            symbols=["AAPL", "GOOGL"],
        )

        assert "AAPL" in results
        assert "GOOGL" in results

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_predictions_history(self, service_with_model):
        """测试预测历史"""
        await service_with_model.start()

        await service_with_model.predict("MODEL001", "AAPL", {"x": 50.0})
        await service_with_model.predict("MODEL001", "GOOGL", {"x": 100.0})

        history = service_with_model.get_predictions_history()
        assert len(history) >= 2

        # 按模型筛选
        model_history = service_with_model.get_predictions_history(model_id="MODEL001")
        assert len(model_history) >= 2

        # 按标的筛选
        symbol_history = service_with_model.get_predictions_history(symbol="AAPL")
        assert len(symbol_history) >= 1

        await service_with_model.stop()


# ============ Signal Generation Tests ============


class TestSignalGeneration:
    """信号生成测试"""

    @pytest.mark.asyncio
    async def test_generate_signal(self, service_with_model):
        """测试生成信号"""
        await service_with_model.start()

        signal = await service_with_model.generate_signal(
            model_id="MODEL001",
            symbol="AAPL",
            current_price=150.0,
        )

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.source_model == "MODEL001"

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_signal_with_callback(self, service_with_model):
        """测试信号回调"""
        signals_received = []

        def callback(signal):
            signals_received.append(signal)

        service_with_model.signal_callback = callback
        await service_with_model.start()

        await service_with_model.generate_signal("MODEL001", "AAPL", 150.0)

        assert len(signals_received) == 1

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_get_signals(self, service_with_model):
        """测试获取信号"""
        await service_with_model.start()

        await service_with_model.generate_signal("MODEL001", "AAPL", 150.0)
        await service_with_model.generate_signal("MODEL001", "GOOGL", 2800.0)

        # 获取所有信号
        all_signals = service_with_model.get_signals()
        assert len(all_signals) >= 2

        # 按标的筛选
        aapl_signals = service_with_model.get_signals(symbol="AAPL")
        assert len(aapl_signals) >= 1

        await service_with_model.stop()

    @pytest.mark.asyncio
    async def test_signal_direction_mapping(self, service_with_model, model_provider):
        """测试信号方向映射"""
        # 设置强多头预测
        def strong_long_fn(model, features):
            return 0.8, 0.9
        model_provider.set_predict_function(strong_long_fn)

        await service_with_model.start()
        signal = await service_with_model.generate_signal("MODEL001", "AAPL", 150.0)

        assert signal.direction == SignalDirection.STRONG_LONG
        assert signal.strength > 0.5

        await service_with_model.stop()


# ============ Cache Management Tests ============


class TestCacheManagement:
    """缓存管理测试"""

    def test_get_cache_stats(self, service):
        """测试获取缓存统计"""
        stats = service.get_cache_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 1000

    def test_clear_cache(self, service):
        """测试清除缓存"""
        # 手动添加缓存
        service._cache["MODEL001:xxx"] = (None, datetime.now())
        service._cache["MODEL002:yyy"] = (None, datetime.now())

        # 清除特定模型缓存
        count = service.clear_cache("MODEL001")
        assert count == 1
        assert "MODEL001:xxx" not in service._cache
        assert "MODEL002:yyy" in service._cache

        # 清除所有缓存
        count = service.clear_cache()
        assert count == 1
        assert len(service._cache) == 0


# ============ Drift Detection Tests ============


class TestDriftDetection:
    """漂移检测测试"""

    def test_set_baseline_metrics(self, service_with_model):
        """测试设置基线指标"""
        service_with_model.set_baseline_metrics("MODEL001", {
            "mean_prediction": 0.5,
            "mean_confidence": 0.8,
        })
        assert "MODEL001" in service_with_model._baseline_metrics

    @pytest.mark.asyncio
    async def test_drift_alert_creation(self, service_with_model):
        """测试漂移告警创建"""
        await service_with_model.start()

        # 设置基线
        service_with_model.set_baseline_metrics("MODEL001", {
            "mean_prediction": 0.5,
        })

        # 生成一些预测以触发漂移检测
        for _ in range(110):
            await service_with_model.predict("MODEL001", "AAPL", {"x": 100.0})  # 极端特征值

        # 等待监控循环运行
        await asyncio.sleep(0.2)

        await service_with_model.stop()

    def test_get_drift_alerts(self, service_with_model):
        """测试获取漂移告警"""
        # 手动添加告警
        alert = DriftAlert(
            alert_id="ALERT001",
            model_id="MODEL001",
            drift_type=DriftType.PREDICTION_DRIFT,
            severity=AlertSeverity.WARNING,
            metric_name="mean_prediction",
            baseline_value=0.5,
            current_value=0.7,
            threshold=0.1,
            description="Test drift",
        )
        service_with_model._alerts.append(alert)

        alerts = service_with_model.get_drift_alerts()
        assert len(alerts) == 1

        # 按模型筛选
        model_alerts = service_with_model.get_drift_alerts(model_id="MODEL001")
        assert len(model_alerts) == 1

        # 按严重程度筛选
        warning_alerts = service_with_model.get_drift_alerts(severity="warning")
        assert len(warning_alerts) == 1

    def test_acknowledge_alert(self, service_with_model):
        """测试确认告警"""
        alert = DriftAlert(
            alert_id="ALERT001",
            model_id="MODEL001",
            drift_type=DriftType.FEATURE_DRIFT,
            severity=AlertSeverity.INFO,
            metric_name="test",
            baseline_value=0.5,
            current_value=0.6,
            threshold=0.1,
            description="Test",
        )
        service_with_model._alerts.append(alert)

        result = service_with_model.acknowledge_alert("ALERT001", "admin")
        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin"

    def test_drift_callback(self, service_with_model):
        """测试漂移回调"""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        service_with_model.drift_callback = callback
        service_with_model._create_drift_alert(
            model_id="MODEL001",
            metric_name="mean_prediction",
            baseline_value=0.5,
            current_value=0.7,
            drift=0.4,
        )

        assert len(alerts_received) == 1


# ============ Performance Metrics Tests ============


class TestPerformanceMetrics:
    """性能指标测试"""

    def test_get_model_performance_no_predictions(self, service_with_model):
        """测试无预测时的性能指标"""
        metrics = service_with_model.get_model_performance("MODEL001")
        assert metrics is not None
        assert metrics.total_predictions == 0

    @pytest.mark.asyncio
    async def test_get_model_performance_with_predictions(self, service_with_model):
        """测试有预测时的性能指标"""
        await service_with_model.start()

        # 使用不同的特征来避免缓存命中
        for i in range(10):
            await service_with_model.predict("MODEL001", "AAPL", {"x": float(i * 10)})

        metrics = service_with_model.get_model_performance("MODEL001")
        assert metrics is not None
        assert metrics.total_predictions == 10
        assert metrics.avg_latency_ms >= 0

        await service_with_model.stop()

    def test_get_nonexistent_model_performance(self, service):
        """测试不存在模型的性能指标"""
        metrics = service.get_model_performance("UNKNOWN")
        assert metrics is None


# ============ Statistics Tests ============


class TestStatistics:
    """统计测试"""

    def test_get_statistics(self, service_with_model):
        """测试获取统计信息"""
        stats = service_with_model.get_statistics()

        assert stats["state"] == "stopped"
        assert stats["registered_models"] == 1
        assert stats["loaded_models"] == 0
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_statistics_after_operations(self, service_with_model):
        """测试操作后的统计"""
        await service_with_model.start()

        await service_with_model.predict("MODEL001", "AAPL", {"x": 50.0})
        await service_with_model.generate_signal("MODEL001", "GOOGL", 150.0)

        stats = service_with_model.get_statistics()
        assert stats["state"] == "running"
        assert stats["total_predictions"] >= 2
        assert stats["total_signals"] >= 1

        await service_with_model.stop()


# ============ Factory Function Tests ============


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_inference_service(self):
        """测试创建推理服务"""
        service = create_inference_service(
            cache_ttl_seconds=120,
            enable_caching=False,
        )
        assert service.config.cache_ttl_seconds == 120
        assert service.config.enable_caching is False

    def test_create_model_metadata(self):
        """测试创建模型元数据"""
        metadata = create_model_metadata(
            model_id="M1",
            name="Test Model",
            model_type="classification",
            version="1.0",
            description="Test description",
        )
        assert metadata.model_id == "M1"
        assert metadata.model_type == ModelType.CLASSIFICATION

    def test_create_model_version(self):
        """测试创建模型版本"""
        version = create_model_version(
            model_id="M1",
            version="1.0",
            is_champion=True,
        )
        assert version.model_id == "M1"
        assert version.version == "1.0"
        assert version.status == ModelStatus.REGISTERED
        assert version.is_champion is True

    @pytest.mark.asyncio
    async def test_quick_predict(self, service_with_model):
        """测试快速预测"""
        await service_with_model.start()

        result = await quick_predict(
            service_with_model,
            model_id="MODEL001",
            symbol="AAPL",
            features={"x": 50.0},
        )

        assert result is not None
        assert result["model_id"] == "MODEL001"
        assert result["symbol"] == "AAPL"

        await service_with_model.stop()


# ============ Integration Tests ============


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建提供者
        feature_provider = SimulatedFeatureProvider()
        model_provider = SimulatedModelProvider()
        model_provider.set_model("SIGNAL_MODEL", "1.0.0", {"type": "signal_model"})

        # 创建服务
        service = MLInferenceService(
            config=InferenceConfig(
                cache_ttl_seconds=60,
                enable_drift_detection=True,
            ),
            feature_provider=feature_provider,
            model_provider=model_provider,
        )

        # 注册模型
        metadata = create_model_metadata(
            model_id="SIGNAL_MODEL",
            name="Trading Signal Model",
            model_type="classification",
            version="1.0.0",
            features=["momentum", "volatility", "trend"],
        )
        service.register_model(metadata)

        # 添加版本
        version = create_model_version("SIGNAL_MODEL", "1.0.0", is_champion=True)
        service.add_model_version(version)

        # 设置基线
        service.set_baseline_metrics("SIGNAL_MODEL", {
            "mean_prediction": 0.0,
            "mean_confidence": 0.7,
        })

        # 启动服务
        await service.start()
        assert service.state == ServiceState.RUNNING

        # 执行预测
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            prediction = await service.predict(
                "SIGNAL_MODEL",
                symbol,
                {"momentum": 0.5 + i * 0.1, "volatility": 0.2, "trend": 0.3},
            )
            assert prediction is not None

        # 生成信号
        signal = await service.generate_signal(
            "SIGNAL_MODEL",
            "AAPL",
            current_price=150.0,
        )
        assert signal is not None

        # 检查统计
        stats = service.get_statistics()
        assert stats["total_predictions"] >= 3
        assert stats["total_signals"] >= 1

        # 检查性能
        performance = service.get_model_performance("SIGNAL_MODEL")
        assert performance.total_predictions >= 3

        # 停止服务
        await service.stop()
        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_multi_model_workflow(self):
        """测试多模型工作流程"""
        # 创建提供者
        model_provider = SimulatedModelProvider()

        service = MLInferenceService(
            config=InferenceConfig(),
            model_provider=model_provider,
        )

        # 注册多个模型
        for i, model_type in enumerate(["classification", "regression", "ranking"]):
            model_id = f"MODEL{i+1}"
            model_provider.set_model(model_id, "1.0.0", {"type": model_type})

            metadata = create_model_metadata(
                model_id=model_id,
                name=f"Model {i+1}",
                model_type=model_type,
                version="1.0.0",
            )
            service.register_model(metadata)

            version = create_model_version(model_id, "1.0.0", is_champion=True)
            service.add_model_version(version)

        await service.start()

        # 从每个模型获取预测
        for i in range(1, 4):
            prediction = await service.predict(f"MODEL{i}", "AAPL", {"x": 50.0 + i})
            assert prediction is not None

        # 检查所有模型都有记录
        stats = service.get_statistics()
        assert stats["registered_models"] == 3
        assert stats["total_predictions"] >= 3

        await service.stop()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """测试错误恢复"""
        model_provider = SimulatedModelProvider()
        model_provider.set_model("TEST", "1.0", {"type": "test"})

        service = MLInferenceService(
            config=InferenceConfig(),
            model_provider=model_provider,
        )

        await service.start()

        # 尝试预测未注册的模型
        prediction = await service.predict("UNKNOWN_MODEL", "AAPL", {"x": 50.0})
        assert prediction is None

        # 服务应该仍在运行
        assert service.state == ServiceState.RUNNING

        # 注册模型后应该可以正常预测
        metadata = create_model_metadata("TEST", "Test", "regression", "1.0")
        service.register_model(metadata)
        version = create_model_version("TEST", "1.0", is_champion=True)
        service.add_model_version(version)

        prediction = await service.predict("TEST", "AAPL", {"x": 50.0})
        assert prediction is not None

        await service.stop()
