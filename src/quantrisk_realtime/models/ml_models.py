# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ML模型模块

提供机器学习模型用于价格预测和信号生成:
- RandomForest: 随机森林分类/回归
- XGBoost: 梯度提升树
- LSTM: 长短期记忆网络 (时间序列预测)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """预测类型"""
    DIRECTION = "direction"      # 方向预测 (上涨/下跌)
    PRICE = "price"              # 价格预测
    RETURN = "return"            # 收益率预测
    VOLATILITY = "volatility"    # 波动率预测


@dataclass
class MLPrediction:
    """ML模型预测结果"""
    model_name: str
    prediction_type: PredictionType
    value: float                      # 预测值
    probability: Optional[float] = None  # 概率 (分类问题)
    confidence: float = 0.5           # 置信度
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_signal(self) -> float:
        """转换为信号 [-1, 1]"""
        if self.prediction_type == PredictionType.DIRECTION:
            # 二分类: 0=下跌, 1=上涨
            prob = self.probability or 0.5
            return (prob - 0.5) * 2 * self.confidence
        elif self.prediction_type == PredictionType.RETURN:
            # 收益率: 正数看多，负数看空
            return np.clip(self.value * 10, -1, 1) * self.confidence
        elif self.prediction_type == PredictionType.VOLATILITY:
            # 波动率高时偏保守
            return -np.clip(self.value / 0.3, 0, 1) * self.confidence * 0.5
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prediction_type": self.prediction_type.value,
            "value": self.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "signal": self.to_signal(),
            "feature_importance": self.feature_importance,
            "timestamp": self.timestamp.isoformat(),
        }


class MLModel(ABC):
    """ML模型基类"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModel":
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> MLPrediction:
        """预测"""
        pass

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        准备特征

        Args:
            df: 原始数据
            feature_columns: 特征列名

        Returns:
            特征DataFrame
        """
        if feature_columns:
            return df[feature_columns].copy()

        # 默认特征
        default_features = [
            "close", "volume", "Vol", "ATR", "RSI", "MACD",
            "MA1", "MA2", "MA3", "MA4", "Roll",
        ]

        available = [f for f in default_features if f in df.columns]
        return df[available].copy()

    def create_labels(
        self,
        df: pd.DataFrame,
        label_type: str = "direction",
        periods: int = 5,
    ) -> pd.Series:
        """
        创建标签

        Args:
            df: 原始数据
            label_type: 标签类型 (direction, return)
            periods: 预测周期

        Returns:
            标签Series
        """
        if label_type == "direction":
            future_return = df["close"].pct_change(periods).shift(-periods)
            return (future_return > 0).astype(int)
        elif label_type == "return":
            return df["close"].pct_change(periods).shift(-periods)
        else:
            raise ValueError(f"Unknown label type: {label_type}")


class RandomForestPredictor(MLModel):
    """随机森林预测器"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        prediction_type: PredictionType = PredictionType.DIRECTION,
    ):
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prediction_type = prediction_type

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPredictor":
        """训练随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # 处理NaN
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index].dropna()
            X_clean = X_clean.loc[y_clean.index]

            if len(X_clean) < 50:
                logger.warning("Not enough data to train RandomForest")
                return self

            self.feature_names = list(X_clean.columns)

            if self.prediction_type == PredictionType.DIRECTION:
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=42,
                    n_jobs=-1,
                )

            self.model.fit(X_clean.values, y_clean.values)
            self.is_fitted = True

            logger.info(f"RandomForest fitted with {len(X_clean)} samples")

        except ImportError:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")

        return self

    def predict(self, X: pd.DataFrame) -> MLPrediction:
        """预测"""
        if not self.is_fitted or self.model is None:
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
            )

        try:
            # 确保特征顺序一致
            X_aligned = X[self.feature_names].iloc[[-1]]  # 取最新一行

            if X_aligned.isna().any().any():
                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "NaN in features"},
                )

            if self.prediction_type == PredictionType.DIRECTION:
                prob = self.model.predict_proba(X_aligned.values)[0]
                pred_class = self.model.predict(X_aligned.values)[0]

                # 特征重要性
                importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))

                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=float(pred_class),
                    probability=float(prob[1]),  # P(上涨)
                    confidence=float(max(prob)),
                    feature_importance=importance,
                )
            else:
                pred = self.model.predict(X_aligned.values)[0]
                importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))

                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=float(pred),
                    confidence=0.7,  # 回归问题置信度固定
                    feature_importance=importance,
                )

        except Exception as e:
            logger.error(f"RandomForest prediction failed: {e}")
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )


class XGBoostPredictor(MLModel):
    """XGBoost预测器"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        prediction_type: PredictionType = PredictionType.DIRECTION,
    ):
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.prediction_type = prediction_type

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostPredictor":
        """训练XGBoost模型"""
        try:
            import xgboost as xgb

            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index].dropna()
            X_clean = X_clean.loc[y_clean.index]

            if len(X_clean) < 50:
                logger.warning("Not enough data to train XGBoost")
                return self

            self.feature_names = list(X_clean.columns)

            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }

            if self.prediction_type == PredictionType.DIRECTION:
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)

            self.model.fit(X_clean.values, y_clean.values)
            self.is_fitted = True

            logger.info(f"XGBoost fitted with {len(X_clean)} samples")

        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

        return self

    def predict(self, X: pd.DataFrame) -> MLPrediction:
        """预测"""
        if not self.is_fitted or self.model is None:
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
            )

        try:
            X_aligned = X[self.feature_names].iloc[[-1]]

            if X_aligned.isna().any().any():
                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=0.0,
                    confidence=0.0,
                )

            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))

            if self.prediction_type == PredictionType.DIRECTION:
                prob = self.model.predict_proba(X_aligned.values)[0]
                pred_class = self.model.predict(X_aligned.values)[0]

                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=float(pred_class),
                    probability=float(prob[1]),
                    confidence=float(max(prob)),
                    feature_importance=importance,
                )
            else:
                pred = self.model.predict(X_aligned.values)[0]

                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=float(pred),
                    confidence=0.7,
                    feature_importance=importance,
                )

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
            )


class LSTMPredictor(MLModel):
    """LSTM时间序列预测器"""

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        prediction_type: PredictionType = PredictionType.RETURN,
    ):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_type = prediction_type
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMPredictor":
        """训练LSTM模型"""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler

            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index].dropna()
            X_clean = X_clean.loc[y_clean.index]

            if len(X_clean) < self.sequence_length + 50:
                logger.warning("Not enough data to train LSTM")
                return self

            self.feature_names = list(X_clean.columns)

            # 标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_clean.values)

            # 创建序列
            X_seq, y_seq = self._create_sequences(X_scaled, y_clean.values)

            if len(X_seq) < 50:
                logger.warning("Not enough sequences to train LSTM")
                return self

            # 定义LSTM模型
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=0.2
                    )
                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])

            input_size = X_clean.shape[1]
            output_size = 1

            self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, output_size)

            # 训练
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            self.model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    logger.debug(f"LSTM Epoch {epoch+1}/100, Loss: {loss.item():.6f}")

            self.model.eval()
            self.is_fitted = True

            logger.info(f"LSTM fitted with {len(X_seq)} sequences")

        except ImportError:
            logger.error("PyTorch not installed. Run: pip install torch")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")

        return self

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列数据"""
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def predict(self, X: pd.DataFrame) -> MLPrediction:
        """预测"""
        if not self.is_fitted or self.model is None or self.scaler is None:
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
            )

        try:
            import torch

            X_aligned = X[self.feature_names].tail(self.sequence_length)

            if len(X_aligned) < self.sequence_length or X_aligned.isna().any().any():
                return MLPrediction(
                    model_name=self.name,
                    prediction_type=self.prediction_type,
                    value=0.0,
                    confidence=0.0,
                )

            X_scaled = self.scaler.transform(X_aligned.values)
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).item()

            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=float(pred),
                confidence=0.6,  # LSTM置信度
            )

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return MLPrediction(
                model_name=self.name,
                prediction_type=self.prediction_type,
                value=0.0,
                confidence=0.0,
            )


# ============ 便捷函数 ============

_ml_models: Dict[str, MLModel] = {}


def get_ml_model(model_type: str = "random_forest", **kwargs) -> MLModel:
    """
    获取或创建ML模型

    Args:
        model_type: 模型类型 (random_forest, xgboost, lstm)
        **kwargs: 模型参数

    Returns:
        MLModel实例
    """
    if model_type not in _ml_models:
        if model_type == "random_forest":
            _ml_models[model_type] = RandomForestPredictor(**kwargs)
        elif model_type == "xgboost":
            _ml_models[model_type] = XGBoostPredictor(**kwargs)
        elif model_type == "lstm":
            _ml_models[model_type] = LSTMPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return _ml_models[model_type]
