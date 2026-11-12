# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
算法模型模块

提供三类模型:
1. ML模型 (RandomForest, XGBoost, LSTM)
2. 统计模型 (GARCH, 协整检验, 卡尔曼滤波)
3. 因子模型 (动量, 价值, 质量)
"""

from .ml_models import (
    MLModel,
    MLPrediction,
    PredictionType,
    RandomForestPredictor,
    XGBoostPredictor,
    LSTMPredictor,
    get_ml_model,
)

from .statistical_models import (
    StatisticalModel,
    StatisticalResult,
    GARCHModel,
    CointegrationModel,
    KalmanFilterModel,
    get_statistical_model,
)

from .factor_models import (
    FactorModel,
    FactorValue,
    FactorExposure,
    MultiFactorResult,
    MomentumFactor,
    ValueFactor,
    QualityFactor,
    MultiFactorModel,
    get_factor_model,
    get_multi_factor_model,
)

# signal_providers (adapters for the host's decision engine V2) was not extracted.

__all__ = [
    # ML Models
    "MLModel",
    "MLPrediction",
    "PredictionType",
    "RandomForestPredictor",
    "XGBoostPredictor",
    "LSTMPredictor",
    "get_ml_model",
    # Statistical Models
    "StatisticalModel",
    "StatisticalResult",
    "GARCHModel",
    "CointegrationModel",
    "KalmanFilterModel",
    "get_statistical_model",
    # Factor Models
    "FactorModel",
    "FactorValue",
    "FactorExposure",
    "MultiFactorResult",
    "MomentumFactor",
    "ValueFactor",
    "QualityFactor",
    "MultiFactorModel",
    "get_factor_model",
    "get_multi_factor_model",
]
