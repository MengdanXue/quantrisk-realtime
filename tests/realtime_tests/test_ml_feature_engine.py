"""
ML Feature Engine Tests - 机器学习特征引擎测试

全面测试特征生成、选择、变换和监控功能。
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.ml_feature_engine import (
    # Enums
    FeatureType,
    TransformMethod,
    SelectionMethod,
    DriftType,
    AlertSeverity,
    EngineState,
    CacheStrategy,
    # Data Classes
    FeatureMetadata,
    FeatureValue,
    FeatureVector,
    FeatureStatistics,
    FeatureDrift,
    FeatureImportance,
    FeatureAlert,
    TransformConfig,
    SelectionResult,
    EngineConfig,
    # Providers
    SimulatedDataProvider,
    InMemoryFeatureStorage,
    # Generators
    TechnicalFeatureGenerator,
    LagFeatureGenerator,
    RollingFeatureGenerator,
    InteractionFeatureGenerator,
    # Transformers and Selectors
    FeatureTransformer,
    FeatureSelector,
    # Monitors
    DriftDetector,
    # Main Class
    MLFeatureEngine,
    # Factory Functions
    create_feature_engine,
    create_feature_metadata,
    quick_feature_generation,
)


# ============ Fixtures ============


@pytest.fixture
def sample_prices():
    """样本价格数据"""
    return [100 + i + np.random.randn() * 2 for i in range(100)]


@pytest.fixture
def sample_volumes():
    """样本成交量数据"""
    return [1000000 + np.random.randint(-100000, 100000) for _ in range(100)]


@pytest.fixture
def data_provider(sample_prices, sample_volumes):
    """数据提供者"""
    provider = SimulatedDataProvider()
    provider.set_price_data("AAPL", sample_prices)
    provider.set_volume_data("AAPL", sample_volumes)
    provider.set_features("AAPL", {"existing_feature": 42.0})
    return provider


@pytest.fixture
def feature_storage():
    """特征存储"""
    return InMemoryFeatureStorage()


@pytest.fixture
def engine_config():
    """引擎配置"""
    return EngineConfig(
        enable_technical_features=True,
        enable_lag_features=True,
        enable_rolling_features=True,
        enable_interaction_features=False,  # 减少测试时间
        enable_drift_detection=True,
        cache_strategy=CacheStrategy.LRU,
    )


@pytest.fixture
async def feature_engine(engine_config, data_provider, feature_storage):
    """特征引擎"""
    engine = MLFeatureEngine(
        config=engine_config,
        data_provider=data_provider,
        feature_storage=feature_storage,
    )
    yield engine
    if engine._running:
        await engine.stop()


# ============ Enum Tests ============


def test_feature_type_enum():
    """测试特征类型枚举"""
    assert FeatureType.TECHNICAL.value == "technical"
    assert FeatureType.LAG.value == "lag"
    assert FeatureType.ROLLING.value == "rolling"
    assert FeatureType.INTERACTION.value == "interaction"


def test_transform_method_enum():
    """测试变换方法枚举"""
    assert TransformMethod.STANDARDIZE.value == "standardize"
    assert TransformMethod.NORMALIZE.value == "normalize"
    assert TransformMethod.LOG.value == "log"
    assert TransformMethod.BOX_COX.value == "box_cox"


def test_selection_method_enum():
    """测试选择方法枚举"""
    assert SelectionMethod.CORRELATION.value == "correlation"
    assert SelectionMethod.IMPORTANCE.value == "importance"
    assert SelectionMethod.RFE.value == "rfe"


def test_drift_type_enum():
    """测试漂移类型枚举"""
    assert DriftType.MEAN_SHIFT.value == "mean_shift"
    assert DriftType.VARIANCE_SHIFT.value == "variance_shift"


# ============ Data Class Tests ============


def test_feature_metadata_creation():
    """测试特征元数据创建"""
    metadata = FeatureMetadata(
        feature_id="feat_001",
        feature_name="RSI",
        feature_type=FeatureType.TECHNICAL,
        description="Relative Strength Index",
        dependencies=["price"],
    )

    assert metadata.feature_id == "feat_001"
    assert metadata.feature_name == "RSI"
    assert metadata.feature_type == FeatureType.TECHNICAL
    assert "price" in metadata.dependencies


def test_feature_metadata_to_dict():
    """测试特征元数据序列化"""
    metadata = FeatureMetadata(
        feature_id="feat_001",
        feature_name="RSI",
        feature_type=FeatureType.TECHNICAL,
        description="Test",
        dependencies=["price"],
    )

    data = metadata.to_dict()
    assert data["feature_id"] == "feat_001"
    assert data["feature_type"] == "technical"
    assert "dependencies" in data


def test_feature_value_creation():
    """测试特征值创建"""
    value = FeatureValue(
        feature_id="feat_001",
        value=42.5,
        timestamp=datetime.now(),
        confidence=0.95,
    )

    assert value.feature_id == "feat_001"
    assert value.value == 42.5
    assert value.confidence == 0.95
    assert value.is_valid


def test_feature_vector_creation():
    """测试特征向量创建"""
    features = {
        "feat_001": FeatureValue("feat_001", 1.0, datetime.now()),
        "feat_002": FeatureValue("feat_002", 2.0, datetime.now()),
    }

    vector = FeatureVector(
        symbol="AAPL",
        features=features,
        timestamp=datetime.now(),
    )

    assert vector.symbol == "AAPL"
    assert len(vector.features) == 2
    assert vector.is_complete


def test_feature_vector_to_array():
    """测试特征向量转数组"""
    features = {
        "feat_001": FeatureValue("feat_001", 1.0, datetime.now()),
        "feat_002": FeatureValue("feat_002", 2.0, datetime.now()),
    }

    vector = FeatureVector("AAPL", features, datetime.now())
    array = vector.to_array(["feat_001", "feat_002", "feat_003"])

    assert len(array) == 3
    assert array[0] == 1.0
    assert array[1] == 2.0
    assert np.isnan(array[2])


def test_feature_statistics_creation():
    """测试特征统计创建"""
    stats = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=50.0,
        std=10.0,
        min=30.0,
        max=70.0,
        q25=45.0,
        q50=50.0,
        q75=55.0,
        missing_rate=0.0,
        outlier_rate=0.05,
    )

    assert stats.count == 100
    assert stats.mean == 50.0
    assert stats.std == 10.0


# ============ Technical Feature Generator Tests ============


def test_generate_returns(sample_prices):
    """测试收益率特征生成"""
    gen = TechnicalFeatureGenerator()
    features = gen.generate_returns(sample_prices, periods=[1, 5, 10])

    assert "return_1d" in features
    assert "return_5d" in features
    assert "return_10d" in features
    assert isinstance(features["return_1d"], float)


def test_generate_returns_insufficient_data():
    """测试收益率生成-数据不足"""
    gen = TechnicalFeatureGenerator()
    features = gen.generate_returns([100.0], periods=[5])

    assert "return_5d" not in features


def test_generate_moving_averages(sample_prices):
    """测试移动平均特征生成"""
    gen = TechnicalFeatureGenerator()
    features = gen.generate_moving_averages(sample_prices, windows=[5, 10, 20])

    assert "ma_5" in features
    assert "ma_10" in features
    assert "ma_20" in features
    assert "price_to_ma_5" in features


def test_generate_volatility(sample_prices):
    """测试波动率特征生成"""
    gen = TechnicalFeatureGenerator()
    features = gen.generate_volatility(sample_prices, windows=[5, 10, 20])

    assert "volatility_5d" in features
    assert "volatility_10d" in features
    assert features["volatility_5d"] > 0


def test_generate_momentum(sample_prices):
    """测试动量特征生成"""
    gen = TechnicalFeatureGenerator()
    features = gen.generate_momentum(sample_prices)

    # RSI需要至少14个数据点
    if len(sample_prices) >= 14:
        assert "rsi_14" in features
        assert 0 <= features["rsi_14"] <= 100

    # MACD需要至少26个数据点
    if len(sample_prices) >= 26:
        assert "macd" in features


# ============ Lag Feature Generator Tests ============


def test_lag_feature_generation():
    """测试滞后特征生成"""
    gen = LagFeatureGenerator()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    features = gen.generate(data, lags=[1, 2, 3])

    assert "lag_1" in features
    assert "lag_2" in features
    assert "lag_3" in features
    assert features["lag_1"] == 4.0
    assert features["lag_2"] == 3.0


def test_lag_feature_insufficient_data():
    """测试滞后特征-数据不足"""
    gen = LagFeatureGenerator()
    data = [1.0, 2.0]
    features = gen.generate(data, lags=[5])

    assert "lag_5" not in features


# ============ Rolling Feature Generator Tests ============


def test_rolling_feature_generation():
    """测试滚动统计特征生成"""
    gen = RollingFeatureGenerator()
    data = list(range(1, 21))
    features = gen.generate(data, windows=[5, 10])

    assert "rolling_mean_5" in features
    assert "rolling_std_5" in features
    assert "rolling_min_5" in features
    assert "rolling_max_5" in features


def test_rolling_feature_with_skew_kurt():
    """测试滚动特征包含偏度和峰度"""
    gen = RollingFeatureGenerator()
    data = list(range(1, 21))
    features = gen.generate(data, windows=[10])

    assert "rolling_skew_10" in features
    assert "rolling_kurt_10" in features


# ============ Interaction Feature Generator Tests ============


def test_interaction_feature_generation():
    """测试交互特征生成"""
    gen = InteractionFeatureGenerator()
    features = {
        "feat_a": 2.0,
        "feat_b": 3.0,
    }

    interaction = gen.generate(features, pairs=[("feat_a", "feat_b")])

    assert "feat_a_x_feat_b" in interaction
    assert "feat_a_div_feat_b" in interaction
    assert "feat_a_minus_feat_b" in interaction
    assert interaction["feat_a_x_feat_b"] == 6.0


def test_interaction_feature_division_by_zero():
    """测试交互特征-除零保护"""
    gen = InteractionFeatureGenerator()
    features = {
        "feat_a": 2.0,
        "feat_b": 0.0,
    }

    interaction = gen.generate(features, pairs=[("feat_a", "feat_b")])

    # 应该没有除法特征
    assert "feat_a_div_feat_b" not in interaction


# ============ Feature Transformer Tests ============


def test_standardize_transform():
    """测试标准化变换"""
    transformer = FeatureTransformer()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    transformer.fit("feat_001", values, TransformMethod.STANDARDIZE)
    transformed = transformer.transform("feat_001", 3.0)

    # 应该接近0（均值）
    assert abs(transformed) < 0.5


def test_normalize_transform():
    """测试归一化变换"""
    transformer = FeatureTransformer()
    values = [0.0, 25.0, 50.0, 75.0, 100.0]

    transformer.fit("feat_001", values, TransformMethod.NORMALIZE)

    transformed_min = transformer.transform("feat_001", 0.0)
    transformed_max = transformer.transform("feat_001", 100.0)
    transformed_mid = transformer.transform("feat_001", 50.0)

    assert abs(transformed_min) < 0.01
    assert abs(transformed_max - 1.0) < 0.01
    assert abs(transformed_mid - 0.5) < 0.01


def test_log_transform():
    """测试对数变换"""
    transformer = FeatureTransformer()
    transformer.fit("feat_001", [1.0, 10.0, 100.0], TransformMethod.LOG)

    transformed = transformer.transform("feat_001", 10.0)
    assert abs(transformed - np.log(10.0)) < 0.01


def test_sqrt_transform():
    """测试平方根变换"""
    transformer = FeatureTransformer()
    transformer.fit("feat_001", [1.0, 4.0, 9.0], TransformMethod.SQRT)

    transformed = transformer.transform("feat_001", 4.0)
    assert abs(transformed - 2.0) < 0.01


# ============ Feature Selector Tests ============


def test_select_by_correlation():
    """测试基于相关性的特征选择"""
    selector = FeatureSelector()

    # 创建高度相关的特征
    base = [float(i) for i in range(10)]
    features = {
        "feat_a": base,
        "feat_b": [x + 0.1 for x in base],  # 高度相关
        "feat_c": [x * 2 for x in base],    # 高度相关
        "feat_d": [10 - x for x in base],   # 负相关但也高相关
    }

    selected = selector.select_by_correlation(features, threshold=0.95)

    # 应该移除一些高度相关的特征
    assert len(selected) < len(features)


def test_select_by_variance():
    """测试基于方差的特征选择"""
    selector = FeatureSelector()

    features = {
        "feat_a": [1.0] * 10,  # 零方差
        "feat_b": [float(i) for i in range(10)],  # 有方差
    }

    selected = selector.select_by_variance(features, threshold=0.01)

    assert "feat_b" in selected
    assert "feat_a" not in selected


def test_select_by_importance():
    """测试基于重要性的特征选择"""
    selector = FeatureSelector()

    importances = {
        "feat_a": 0.5,
        "feat_b": 0.3,
        "feat_c": 0.8,
        "feat_d": 0.1,
    }

    selected = selector.select_by_importance(importances, top_k=2)

    assert len(selected) == 2
    assert "feat_c" in selected
    assert "feat_a" in selected


# ============ Drift Detector Tests ============


def test_detect_mean_shift():
    """测试均值漂移检测"""
    baseline = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=50.0,
        std=10.0,
        min=30.0,
        max=70.0,
        q25=45.0,
        q50=50.0,
        q75=55.0,
        missing_rate=0.0,
        outlier_rate=0.0,
    )

    current = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=55.0,  # 漂移了
        std=10.0,
        min=35.0,
        max=75.0,
        q25=50.0,
        q50=55.0,
        q75=60.0,
        missing_rate=0.0,
        outlier_rate=0.0,
    )

    is_drifted, score = DriftDetector.detect_mean_shift(baseline, current, threshold=0.1)

    assert isinstance(is_drifted, bool)
    assert score > 0


def test_detect_variance_shift():
    """测试方差漂移检测"""
    baseline = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=50.0,
        std=10.0,
        min=30.0,
        max=70.0,
        q25=45.0,
        q50=50.0,
        q75=55.0,
        missing_rate=0.0,
        outlier_rate=0.0,
    )

    current = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=50.0,
        std=20.0,  # 方差增大
        min=10.0,
        max=90.0,
        q25=35.0,
        q50=50.0,
        q75=65.0,
        missing_rate=0.0,
        outlier_rate=0.0,
    )

    is_drifted, score = DriftDetector.detect_variance_shift(baseline, current, threshold=0.5)

    assert isinstance(is_drifted, bool)
    assert score > 0


def test_detect_distribution_shift():
    """测试分布漂移检测"""
    baseline_values = [float(i) for i in range(100)]
    current_values = [float(i) + 50 for i in range(100)]  # 分布移动

    is_drifted, statistic = DriftDetector.detect_distribution_shift(
        baseline_values,
        current_values,
        threshold=0.05
    )

    assert bool(is_drifted)  # Convert numpy bool to Python bool
    assert statistic >= 0


# ============ Engine Lifecycle Tests ============


@pytest.mark.asyncio
async def test_engine_creation(feature_engine):
    """测试引擎创建"""
    assert feature_engine._state == EngineState.IDLE
    assert not feature_engine._running


@pytest.mark.asyncio
async def test_engine_start_stop(feature_engine):
    """测试引擎启动和停止"""
    await feature_engine.start()
    assert feature_engine._running
    assert feature_engine._state == EngineState.RUNNING

    await feature_engine.stop()
    assert not feature_engine._running
    assert feature_engine._state == EngineState.IDLE


@pytest.mark.asyncio
async def test_engine_double_start(feature_engine):
    """测试引擎重复启动"""
    await feature_engine.start()
    await feature_engine.start()  # 应该不报错

    assert feature_engine._running


# ============ Feature Generation Tests ============


@pytest.mark.asyncio
async def test_generate_features(feature_engine):
    """测试特征生成"""
    vector = await feature_engine.generate_features("AAPL")

    assert vector.symbol == "AAPL"
    assert len(vector.features) > 0
    assert vector.is_complete


@pytest.mark.asyncio
async def test_generate_features_with_technical(feature_engine):
    """测试生成技术指标特征"""
    vector = await feature_engine.generate_features("AAPL")

    # 检查是否包含技术指标特征
    tech_features = [k for k in vector.features.keys() if k.startswith("tech_")]
    assert len(tech_features) > 0


@pytest.mark.asyncio
async def test_generate_features_with_lag(feature_engine):
    """测试生成滞后特征"""
    vector = await feature_engine.generate_features("AAPL")

    # 检查是否包含滞后特征
    lag_features = [k for k in vector.features.keys() if "lag_" in k]
    assert len(lag_features) > 0


@pytest.mark.asyncio
async def test_generate_features_with_rolling(feature_engine):
    """测试生成滚动统计特征"""
    vector = await feature_engine.generate_features("AAPL")

    # 检查是否包含滚动特征
    rolling_features = [k for k in vector.features.keys() if "rolling_" in k]
    assert len(rolling_features) > 0


# ============ Feature Registration Tests ============


def test_register_feature(feature_engine):
    """测试特征注册"""
    metadata = FeatureMetadata(
        feature_id="custom_001",
        feature_name="Custom Feature",
        feature_type=FeatureType.CUSTOM,
        description="Test feature",
        dependencies=[],
    )

    feature_engine.register_feature(metadata)

    retrieved = feature_engine.get_feature_metadata("custom_001")
    assert retrieved is not None
    assert retrieved.feature_name == "Custom Feature"


def test_list_features(feature_engine):
    """测试列出特征"""
    metadata1 = FeatureMetadata("feat_001", "F1", FeatureType.TECHNICAL, "Test 1", [])
    metadata2 = FeatureMetadata("feat_002", "F2", FeatureType.LAG, "Test 2", [])

    feature_engine.register_feature(metadata1)
    feature_engine.register_feature(metadata2)

    features = feature_engine.list_features()
    assert len(features) >= 2


# ============ Feature Transformation Tests ============


def test_fit_transformer(feature_engine):
    """测试拟合变换器"""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    feature_engine.fit_transformer("feat_001", values, TransformMethod.STANDARDIZE)

    transformed = feature_engine.transform_feature("feat_001", 3.0)
    assert isinstance(transformed, float)


def test_transform_without_fit(feature_engine):
    """测试未拟合的变换"""
    # 应该返回原值
    result = feature_engine.transform_feature("unknown_feat", 42.0)
    assert result == 42.0


# ============ Feature Selection Tests ============


@pytest.mark.asyncio
async def test_select_features_by_correlation(feature_engine):
    """测试基于相关性的特征选择"""
    base = [float(i) for i in range(10)]
    feature_data = {
        "feat_a": base,
        "feat_b": [x + 0.1 for x in base],
    }

    result = await feature_engine.select_features(
        SelectionMethod.CORRELATION,
        feature_data,
        threshold=0.95
    )

    assert result.method == SelectionMethod.CORRELATION
    assert len(result.selected_features) > 0


@pytest.mark.asyncio
async def test_select_features_by_variance(feature_engine):
    """测试基于方差的特征选择"""
    feature_data = {
        "feat_a": [1.0] * 10,
        "feat_b": [float(i) for i in range(10)],
    }

    result = await feature_engine.select_features(
        SelectionMethod.VARIANCE,
        feature_data,
        threshold=0.01
    )

    assert "feat_b" in result.selected_features


# ============ Feature Importance Tests ============


def test_set_feature_importance(feature_engine):
    """测试设置特征重要性"""
    feature_engine.set_feature_importance("feat_001", 0.8, "manual")

    importance = feature_engine.get_feature_importance("feat_001")
    assert importance is not None
    assert importance.importance_score == 0.8


def test_get_top_features(feature_engine):
    """测试获取最重要特征"""
    feature_engine.set_feature_importance("feat_001", 0.8)
    feature_engine.set_feature_importance("feat_002", 0.5)
    feature_engine.set_feature_importance("feat_003", 0.9)

    top_features = feature_engine.get_top_features(top_k=2)

    assert len(top_features) == 2
    assert top_features[0].importance_score >= top_features[1].importance_score


# ============ Statistics and Monitoring Tests ============


@pytest.mark.asyncio
async def test_feature_statistics_calculation(feature_engine):
    """测试特征统计计算"""
    # 手动添加足够的历史数据来触发统计计算
    from collections import deque

    # 添加一个特征的历史数据
    feature_id = "test_feat_001"
    feature_engine._feature_history[feature_id] = deque([float(i) for i in range(20)])

    # 手动触发统计计算
    values = list(feature_engine._feature_history[feature_id])
    stats = feature_engine._calculate_statistics(feature_id, values)
    feature_engine._statistics[feature_id] = stats

    # 检查统计是否被计算
    assert len(feature_engine._statistics) > 0
    assert feature_engine._statistics[feature_id].count == 20


def test_get_feature_statistics(feature_engine):
    """测试获取特征统计"""
    # 手动添加统计
    stats = FeatureStatistics(
        feature_id="feat_001",
        count=100,
        mean=50.0,
        std=10.0,
        min=30.0,
        max=70.0,
        q25=45.0,
        q50=50.0,
        q75=55.0,
        missing_rate=0.0,
        outlier_rate=0.05,
    )

    feature_engine._statistics["feat_001"] = stats

    retrieved = feature_engine.get_feature_statistics("feat_001")
    assert retrieved is not None
    assert retrieved.mean == 50.0


# ============ Cache Management Tests ============


@pytest.mark.asyncio
async def test_cache_hit(feature_engine):
    """测试缓存命中"""
    # 第一次生成
    vector1 = await feature_engine.generate_features("AAPL")

    # 第二次应该从缓存获取
    vector2 = await feature_engine.generate_features("AAPL")

    assert feature_engine._cache_hits > 0


def test_clear_cache(feature_engine):
    """测试清空缓存"""
    feature_engine._cache["AAPL"] = (None, datetime.now())
    feature_engine._cache_hits = 10

    feature_engine.clear_cache()

    assert len(feature_engine._cache) == 0
    assert feature_engine._cache_hits == 0


# ============ History Management Tests ============


@pytest.mark.asyncio
async def test_feature_history(feature_engine):
    """测试特征历史"""
    # 生成多次特征
    for _ in range(5):
        await feature_engine.generate_features("AAPL")

    # 获取某个特征的历史
    if feature_engine._feature_history:
        feature_id = list(feature_engine._feature_history.keys())[0]
        history = feature_engine.get_feature_history(feature_id, limit=10)
        assert len(history) > 0


def test_get_feature_history_limit(feature_engine):
    """测试特征历史限制"""
    # 手动添加历史
    from collections import deque
    feature_engine._feature_history["feat_001"] = deque([float(i) for i in range(100)])

    history = feature_engine.get_feature_history("feat_001", limit=10)
    assert len(history) == 10


# ============ Alert Tests ============


def test_create_alert(feature_engine):
    """测试创建警报"""
    feature_engine._create_alert(
        "feat_001",
        AlertSeverity.HIGH,
        "Test alert",
        {"test": "data"}
    )

    alerts = feature_engine.get_alerts()
    assert len(alerts) > 0
    assert alerts[0].feature_id == "feat_001"


def test_get_alerts_limit(feature_engine):
    """测试获取警报限制"""
    for i in range(10):
        feature_engine._create_alert(
            f"feat_{i}",
            AlertSeverity.LOW,
            f"Alert {i}",
            {}
        )

    alerts = feature_engine.get_alerts(limit=5)
    assert len(alerts) == 5


# ============ Callback Tests ============


@pytest.mark.asyncio
async def test_feature_update_callback(feature_engine):
    """测试特征更新回调"""
    callback_called = False
    received_vector = None

    def callback(vector):
        nonlocal callback_called, received_vector
        callback_called = True
        received_vector = vector

    feature_engine.on_features_update(callback)

    await feature_engine.generate_features("AAPL")

    assert callback_called
    assert received_vector is not None


def test_alert_callback(feature_engine):
    """测试警报回调"""
    callback_called = False
    received_alert = None

    def callback(alert):
        nonlocal callback_called, received_alert
        callback_called = True
        received_alert = alert

    feature_engine.on_alert(callback)

    feature_engine._create_alert(
        "feat_001",
        AlertSeverity.HIGH,
        "Test",
        {}
    )

    assert callback_called
    assert received_alert is not None


def test_drift_callback(feature_engine):
    """测试漂移检测回调"""
    callback_called = False
    received_drift = None

    def callback(drift):
        nonlocal callback_called, received_drift
        callback_called = True
        received_drift = drift

    feature_engine.on_drift(callback)

    # 手动触发漂移通知
    baseline = FeatureStatistics("feat_001", 100, 50.0, 10.0, 30.0, 70.0, 45.0, 50.0, 55.0, 0.0, 0.0)
    current = FeatureStatistics("feat_001", 100, 60.0, 10.0, 40.0, 80.0, 55.0, 60.0, 65.0, 0.0, 0.0)

    drift = FeatureDrift(
        feature_id="feat_001",
        drift_type=DriftType.MEAN_SHIFT,
        drift_score=1.0,
        baseline_stats=baseline,
        current_stats=current,
        is_drifted=True,
        threshold=0.1,
    )

    feature_engine._notify_drift(drift)

    assert callback_called
    assert received_drift is not None


# ============ Public API Tests ============


def test_get_state(feature_engine):
    """测试获取引擎状态"""
    state = feature_engine.get_state()
    assert state == EngineState.IDLE


def test_get_statistics(feature_engine):
    """测试获取统计信息"""
    stats = feature_engine.get_statistics()

    assert "state" in stats
    assert "features_registered" in stats
    assert "cache_hit_rate" in stats


# ============ Factory Function Tests ============


def test_create_feature_engine():
    """测试创建特征引擎"""
    engine = create_feature_engine(
        enable_technical_features=True,
        enable_lag_features=True,
    )

    assert engine is not None
    assert engine._config.enable_technical_features


def test_create_feature_metadata_factory():
    """测试创建特征元数据工厂函数"""
    metadata = create_feature_metadata(
        feature_id="feat_001",
        feature_name="Test Feature",
        feature_type="technical",
        description="Test",
    )

    assert metadata.feature_id == "feat_001"
    assert metadata.feature_type == FeatureType.TECHNICAL


@pytest.mark.asyncio
async def test_quick_feature_generation():
    """测试快速特征生成"""
    provider = SimulatedDataProvider()
    provider.set_price_data("AAPL", [100 + i for i in range(50)])
    provider.set_volume_data("AAPL", [1000000] * 50)

    engine = create_feature_engine(data_provider=provider)

    vector = await quick_feature_generation(engine, "AAPL")

    assert vector.symbol == "AAPL"
    assert len(vector.features) > 0


# ============ Integration Tests ============


@pytest.mark.asyncio
async def test_end_to_end_feature_pipeline(feature_engine):
    """测试端到端特征管道"""
    # 1. 启动引擎
    await feature_engine.start()

    # 2. 注册特征元数据
    metadata = FeatureMetadata(
        feature_id="custom_001",
        feature_name="Custom",
        feature_type=FeatureType.CUSTOM,
        description="Test",
        dependencies=[],
    )
    feature_engine.register_feature(metadata)

    # 3. 生成特征
    vector = await feature_engine.generate_features("AAPL")

    # 4. 设置重要性
    for feature_id in list(vector.features.keys())[:5]:
        feature_engine.set_feature_importance(feature_id, np.random.random())

    # 5. 获取统计
    stats = feature_engine.get_statistics()

    # 6. 停止引擎
    await feature_engine.stop()

    assert vector is not None
    assert stats["features_registered"] >= 1


@pytest.mark.asyncio
async def test_feature_storage_integration(feature_engine, feature_storage):
    """测试特征存储集成"""
    # 生成特征
    vector = await feature_engine.generate_features("AAPL")

    # 从存储中加载
    loaded = await feature_storage.load_feature_vector("AAPL", datetime.now())

    assert loaded is not None
    assert loaded.symbol == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
