# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
统计模型模块

提供统计模型用于风险建模和配对交易:
- GARCH: 波动率建模
- 协整检验: 配对交易机会检测
- 卡尔曼滤波: 动态对冲比率估计
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """统计模型结果"""
    model_name: str
    value: float                          # 主要输出值
    confidence: float = 0.5               # 置信度
    statistics: Dict[str, float] = field(default_factory=dict)
    is_significant: bool = False          # 统计显著性
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "value": self.value,
            "confidence": self.confidence,
            "statistics": self.statistics,
            "is_significant": self.is_significant,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class StatisticalModel(ABC):
    """统计模型基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> "StatisticalModel":
        """拟合模型"""
        pass

    @abstractmethod
    def predict(self, **kwargs) -> StatisticalResult:
        """预测/推断"""
        pass


class GARCHModel(StatisticalModel):
    """
    GARCH模型 - 波动率预测

    GARCH(1,1)模型:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    用途:
    - 波动率预测
    - 风险度量 (VaR)
    - 期权定价
    """

    def __init__(self, p: int = 1, q: int = 1):
        super().__init__("garch")
        self.p = p
        self.q = q
        self.model = None
        self.fitted_result = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, return_column: str = "returns") -> "GARCHModel":
        """
        拟合GARCH模型

        Args:
            data: 包含收益率的DataFrame
            return_column: 收益率列名

        Returns:
            self
        """
        try:
            from arch import arch_model

            if return_column not in data.columns:
                # 从价格计算收益率
                if "close" in data.columns:
                    returns = data["close"].pct_change().dropna() * 100  # 百分比
                else:
                    logger.error("No return data available")
                    return self
            else:
                returns = data[return_column].dropna() * 100

            if len(returns) < 100:
                logger.warning("Not enough data for GARCH model")
                return self

            # 拟合GARCH(p, q)
            self.model = arch_model(
                returns,
                vol="Garch",
                p=self.p,
                q=self.q,
                rescale=False,
            )

            self.fitted_result = self.model.fit(disp="off")
            self.is_fitted = True

            logger.info(f"GARCH({self.p},{self.q}) fitted successfully")

        except ImportError:
            logger.error("arch package not installed. Run: pip install arch")
        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")

        return self

    def predict(self, horizon: int = 5, **kwargs) -> StatisticalResult:
        """
        预测波动率

        Args:
            horizon: 预测期数

        Returns:
            波动率预测结果
        """
        if not self.is_fitted or self.fitted_result is None:
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )

        try:
            # 预测
            forecast = self.fitted_result.forecast(horizon=horizon)
            variance_forecast = forecast.variance.iloc[-1]

            # 转换为年化波动率
            avg_variance = variance_forecast.mean()
            annual_vol = np.sqrt(avg_variance * 252) / 100  # 除以100因为拟合时乘了100

            # 提取模型参数
            params = self.fitted_result.params

            return StatisticalResult(
                model_name=self.name,
                value=annual_vol,
                confidence=0.7,
                statistics={
                    "omega": float(params.get("omega", 0)),
                    "alpha": float(params.get("alpha[1]", 0)),
                    "beta": float(params.get("beta[1]", 0)),
                    "persistence": float(params.get("alpha[1]", 0) + params.get("beta[1]", 0)),
                    "aic": float(self.fitted_result.aic),
                    "bic": float(self.fitted_result.bic),
                },
                is_significant=True,
                metadata={
                    "horizon": horizon,
                    "variance_forecast": variance_forecast.tolist(),
                },
            )

        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )

    def calculate_var(
        self,
        position_value: float,
        confidence_level: float = 0.95,
    ) -> float:
        """
        计算基于GARCH的VaR

        Args:
            position_value: 持仓价值
            confidence_level: 置信水平

        Returns:
            VaR值
        """
        if not self.is_fitted or self.fitted_result is None:
            return 0.0

        try:
            from scipy.stats import norm

            # 获取条件波动率
            cond_vol = self.fitted_result.conditional_volatility.iloc[-1] / 100

            # 正态分布分位数
            z = norm.ppf(1 - confidence_level)

            # VaR
            var = position_value * cond_vol * abs(z)

            return var

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return 0.0


class CointegrationModel(StatisticalModel):
    """
    协整检验模型 - 配对交易

    用于检测两个时间序列之间的长期均衡关系。
    如果两个序列协整，可以构建配对交易策略。

    方法:
    - Engle-Granger两步法
    - Johansen检验
    """

    def __init__(self, method: str = "engle_granger"):
        super().__init__("cointegration")
        self.method = method
        self.beta = None         # 对冲比率
        self.spread_mean = None  # 价差均值
        self.spread_std = None   # 价差标准差
        self.is_cointegrated = False

    def fit(
        self,
        data: pd.DataFrame,
        series1: str,
        series2: str,
    ) -> "CointegrationModel":
        """
        拟合协整模型

        Args:
            data: 包含两个序列的DataFrame
            series1: 第一个序列列名 (dependent)
            series2: 第二个序列列名 (independent)

        Returns:
            self
        """
        try:
            from statsmodels.tsa.stattools import coint, adfuller
            import statsmodels.api as sm

            y = data[series1].dropna()
            x = data[series2].dropna()

            # 对齐索引
            common_idx = y.index.intersection(x.index)
            y = y.loc[common_idx]
            x = x.loc[common_idx]

            if len(y) < 100:
                logger.warning("Not enough data for cointegration test")
                return self

            if self.method == "engle_granger":
                # Engle-Granger两步法

                # Step 1: 检验协整
                score, pvalue, _ = coint(y, x)

                self.is_cointegrated = pvalue < 0.05

                # Step 2: 估计对冲比率
                x_with_const = sm.add_constant(x)
                model = sm.OLS(y, x_with_const)
                results = model.fit()

                self.beta = results.params.iloc[1]

                # 计算价差
                spread = y - self.beta * x
                self.spread_mean = spread.mean()
                self.spread_std = spread.std()

                logger.info(
                    f"Cointegration test: p-value={pvalue:.4f}, "
                    f"cointegrated={self.is_cointegrated}, beta={self.beta:.4f}"
                )

        except ImportError:
            logger.error("statsmodels not installed. Run: pip install statsmodels")
        except Exception as e:
            logger.error(f"Cointegration fitting failed: {e}")

        return self

    def predict(
        self,
        y_current: float,
        x_current: float,
        **kwargs
    ) -> StatisticalResult:
        """
        预测价差状态和交易信号

        Args:
            y_current: 序列1当前值
            x_current: 序列2当前值

        Returns:
            交易信号结果
        """
        if self.beta is None:
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )

        try:
            # 计算当前价差
            current_spread = y_current - self.beta * x_current

            # Z-score
            if self.spread_std and self.spread_std > 0:
                z_score = (current_spread - self.spread_mean) / self.spread_std
            else:
                z_score = 0.0

            # 交易信号
            # z_score > 2: 价差过高，做空价差 (卖y买x)
            # z_score < -2: 价差过低，做多价差 (买y卖x)
            signal = 0.0
            if z_score > 2:
                signal = -1.0  # 做空价差
            elif z_score < -2:
                signal = 1.0   # 做多价差
            elif abs(z_score) < 0.5:
                signal = 0.0   # 中性区域

            return StatisticalResult(
                model_name=self.name,
                value=signal,
                confidence=min(0.9, abs(z_score) / 3),
                statistics={
                    "beta": float(self.beta),
                    "spread": float(current_spread),
                    "spread_mean": float(self.spread_mean),
                    "spread_std": float(self.spread_std),
                    "z_score": float(z_score),
                },
                is_significant=self.is_cointegrated,
                metadata={
                    "y_current": y_current,
                    "x_current": x_current,
                },
            )

        except Exception as e:
            logger.error(f"Cointegration prediction failed: {e}")
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )


class KalmanFilterModel(StatisticalModel):
    """
    卡尔曼滤波模型 - 动态对冲比率

    用于估计时变的对冲比率，比静态回归更灵活。

    状态空间模型:
    观测方程: y_t = β_t * x_t + ε_t
    状态方程: β_t = β_{t-1} + η_t

    用途:
    - 动态对冲比率估计
    - 配对交易优化
    - 趋势跟踪
    """

    def __init__(
        self,
        delta: float = 1e-4,  # 状态转移噪声
        observation_noise: float = 1.0,
    ):
        super().__init__("kalman_filter")
        self.delta = delta
        self.observation_noise = observation_noise

        # 卡尔曼滤波状态
        self.beta = None           # 当前beta估计
        self.P = None              # 估计误差协方差
        self.beta_history = []     # beta历史

    def fit(
        self,
        data: pd.DataFrame,
        y_column: str,
        x_column: str,
    ) -> "KalmanFilterModel":
        """
        运行卡尔曼滤波

        Args:
            data: 数据
            y_column: 因变量列名
            x_column: 自变量列名

        Returns:
            self
        """
        try:
            y = data[y_column].values
            x = data[x_column].values

            n = len(y)
            if n < 50:
                logger.warning("Not enough data for Kalman filter")
                return self

            # 初始化
            self.beta = 0.0
            self.P = 1.0
            R = self.observation_noise  # 观测噪声
            Q = self.delta              # 状态噪声

            self.beta_history = []

            for t in range(n):
                # 预测步
                beta_pred = self.beta
                P_pred = self.P + Q

                # 更新步
                if x[t] != 0:
                    # 卡尔曼增益
                    K = P_pred * x[t] / (x[t]**2 * P_pred + R)

                    # 更新beta
                    innovation = y[t] - beta_pred * x[t]
                    self.beta = beta_pred + K * innovation

                    # 更新协方差
                    self.P = (1 - K * x[t]) * P_pred

                self.beta_history.append(self.beta)

            logger.info(f"Kalman filter completed, final beta={self.beta:.4f}")

        except Exception as e:
            logger.error(f"Kalman filter failed: {e}")

        return self

    def predict(
        self,
        y_new: float = None,
        x_new: float = None,
        **kwargs
    ) -> StatisticalResult:
        """
        获取当前beta估计或更新

        Args:
            y_new: 新的y观测值 (可选)
            x_new: 新的x观测值 (可选)

        Returns:
            当前beta估计
        """
        if self.beta is None:
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )

        try:
            # 如果有新数据，执行一步更新
            if y_new is not None and x_new is not None and x_new != 0:
                Q = self.delta
                R = self.observation_noise

                # 预测
                beta_pred = self.beta
                P_pred = self.P + Q

                # 更新
                K = P_pred * x_new / (x_new**2 * P_pred + R)
                innovation = y_new - beta_pred * x_new
                self.beta = beta_pred + K * innovation
                self.P = (1 - K * x_new) * P_pred

                self.beta_history.append(self.beta)

            # 计算beta的标准误差
            beta_std = np.sqrt(self.P) if self.P > 0 else 0.0

            return StatisticalResult(
                model_name=self.name,
                value=float(self.beta),
                confidence=max(0, 1 - beta_std),  # 标准误差越小置信度越高
                statistics={
                    "beta": float(self.beta),
                    "beta_std": float(beta_std),
                    "P": float(self.P),
                    "history_length": len(self.beta_history),
                },
                is_significant=True,
            )

        except Exception as e:
            logger.error(f"Kalman filter prediction failed: {e}")
            return StatisticalResult(
                model_name=self.name,
                value=0.0,
                confidence=0.0,
            )


# ============ 便捷函数 ============

_statistical_models: Dict[str, StatisticalModel] = {}


def get_statistical_model(model_type: str = "garch", **kwargs) -> StatisticalModel:
    """
    获取或创建统计模型

    Args:
        model_type: 模型类型 (garch, cointegration, kalman)
        **kwargs: 模型参数

    Returns:
        StatisticalModel实例
    """
    if model_type not in _statistical_models:
        if model_type == "garch":
            _statistical_models[model_type] = GARCHModel(**kwargs)
        elif model_type == "cointegration":
            _statistical_models[model_type] = CointegrationModel(**kwargs)
        elif model_type == "kalman":
            _statistical_models[model_type] = KalmanFilterModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return _statistical_models[model_type]
