# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
因子模型模块

提供量化因子模型:
- MomentumFactor: 动量因子 (价格动量、收益率动量)
- ValueFactor: 价值因子 (PE、PB、股息率)
- QualityFactor: 质量因子 (ROE、负债率、盈利稳定性)
- MultiFactorModel: 多因子组合模型
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
class FactorValue:
    """因子值"""
    factor_name: str
    symbol: str
    value: float                  # 因子值
    z_score: float = 0.0          # 横截面Z-score
    percentile: float = 50.0      # 百分位排名
    signal: float = 0.0           # 转换后的信号 [-1, 1]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "symbol": self.symbol,
            "value": self.value,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "signal": self.signal,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FactorExposure:
    """因子暴露"""
    factor_name: str
    exposure: float               # 暴露度
    t_stat: float = 0.0           # t统计量
    is_significant: bool = False  # 是否显著


@dataclass
class MultiFactorResult:
    """多因子模型结果"""
    symbol: str
    composite_score: float         # 综合得分
    composite_signal: float        # 综合信号 [-1, 1]
    factor_values: List[FactorValue]
    factor_weights: Dict[str, float]
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "composite_score": self.composite_score,
            "composite_signal": self.composite_signal,
            "factor_values": [f.to_dict() for f in self.factor_values],
            "factor_weights": self.factor_weights,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class FactorModel(ABC):
    """因子模型基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate(
        self,
        data: pd.DataFrame,
        symbol: str,
        **kwargs
    ) -> FactorValue:
        """计算因子值"""
        pass

    def normalize_cross_sectional(
        self,
        values: Dict[str, float],
    ) -> Dict[str, Tuple[float, float]]:
        """
        横截面标准化

        Args:
            values: {symbol: factor_value}

        Returns:
            {symbol: (z_score, percentile)}
        """
        if not values:
            return {}

        arr = np.array(list(values.values()))
        mean = np.mean(arr)
        std = np.std(arr)

        result = {}
        for symbol, val in values.items():
            z_score = (val - mean) / std if std > 0 else 0.0
            percentile = (np.sum(arr <= val) / len(arr)) * 100
            result[symbol] = (z_score, percentile)

        return result


class MomentumFactor(FactorModel):
    """
    动量因子

    计算方法:
    - 价格动量: 过去N个月的累计收益率
    - 短期动量: 过去1个月的收益率
    - 动量质量: 动量的稳定性 (低波动率动量)
    """

    def __init__(
        self,
        lookback_periods: int = 252,  # 约1年
        skip_periods: int = 21,        # 跳过最近1个月 (反转效应)
        weight: float = 1.0,
    ):
        super().__init__("momentum", weight)
        self.lookback = lookback_periods
        self.skip = skip_periods

    def calculate(
        self,
        data: pd.DataFrame,
        symbol: str,
        **kwargs
    ) -> FactorValue:
        """
        计算动量因子

        Args:
            data: 包含价格数据的DataFrame
            symbol: 标的代码

        Returns:
            动量因子值
        """
        try:
            close = data["close"] if "close" in data.columns else data.iloc[:, 0]

            if len(close) < self.lookback + self.skip:
                return FactorValue(
                    factor_name=self.name,
                    symbol=symbol,
                    value=0.0,
                    signal=0.0,
                )

            # 计算动量 (跳过最近skip期)
            start_price = close.iloc[-(self.lookback + self.skip)]
            end_price = close.iloc[-self.skip] if self.skip > 0 else close.iloc[-1]

            momentum = (end_price / start_price - 1) * 100  # 百分比

            # 短期动量 (最近1个月)
            short_momentum = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0

            # 动量质量 (低波动率动量更好)
            returns = close.pct_change()
            vol = returns.iloc[-self.lookback:].std() * np.sqrt(252)

            # 风险调整动量
            risk_adj_momentum = momentum / vol if vol > 0 else 0

            # 转换为信号 [-1, 1]
            signal = np.clip(risk_adj_momentum / 5, -1, 1)

            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=momentum,
                signal=signal,
                metadata={
                    "short_momentum": short_momentum,
                    "volatility": vol,
                    "risk_adj_momentum": risk_adj_momentum,
                    "lookback": self.lookback,
                    "skip": self.skip,
                },
            )

        except Exception as e:
            logger.error(f"Momentum calculation failed for {symbol}: {e}")
            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=0.0,
                signal=0.0,
            )


class ValueFactor(FactorModel):
    """
    价值因子

    计算方法:
    - PE Ratio: 市盈率 (低PE看多)
    - PB Ratio: 市净率 (低PB看多)
    - Dividend Yield: 股息率 (高股息看多)
    - Composite Value: 综合价值得分
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("value", weight)

    def calculate(
        self,
        data: pd.DataFrame,
        symbol: str,
        fundamentals: Dict[str, float] = None,
        **kwargs
    ) -> FactorValue:
        """
        计算价值因子

        Args:
            data: 价格数据
            symbol: 标的代码
            fundamentals: 基本面数据 {pe, pb, dividend_yield, ...}

        Returns:
            价值因子值
        """
        try:
            if not fundamentals:
                return FactorValue(
                    factor_name=self.name,
                    symbol=symbol,
                    value=0.0,
                    signal=0.0,
                    metadata={"error": "No fundamental data"},
                )

            pe = fundamentals.get("pe", 0)
            pb = fundamentals.get("pb", 0)
            div_yield = fundamentals.get("dividend_yield", 0)

            # 计算各因子得分 (低估值=高得分)
            scores = []

            if pe > 0:
                # PE越低越好，取倒数并标准化
                pe_score = np.clip(1 / pe * 10, 0, 1)
                scores.append(pe_score)

            if pb > 0:
                # PB越低越好
                pb_score = np.clip(1 / pb * 2, 0, 1)
                scores.append(pb_score)

            if div_yield > 0:
                # 股息率越高越好
                div_score = np.clip(div_yield / 5, 0, 1)  # 5%股息率=满分
                scores.append(div_score)

            if not scores:
                value = 0.0
                signal = 0.0
            else:
                value = np.mean(scores) * 100  # 转换为0-100
                signal = np.clip((value - 50) / 50, -1, 1)  # 转换为[-1, 1]

            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=value,
                signal=signal,
                metadata={
                    "pe": pe,
                    "pb": pb,
                    "dividend_yield": div_yield,
                    "scores": scores,
                },
            )

        except Exception as e:
            logger.error(f"Value calculation failed for {symbol}: {e}")
            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=0.0,
                signal=0.0,
            )


class QualityFactor(FactorModel):
    """
    质量因子

    计算方法:
    - ROE: 股本回报率 (高ROE看多)
    - Debt/Equity: 负债率 (低负债看多)
    - Earnings Stability: 盈利稳定性
    - Gross Margin: 毛利率
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("quality", weight)

    def calculate(
        self,
        data: pd.DataFrame,
        symbol: str,
        fundamentals: Dict[str, float] = None,
        **kwargs
    ) -> FactorValue:
        """
        计算质量因子

        Args:
            data: 价格数据
            symbol: 标的代码
            fundamentals: 基本面数据

        Returns:
            质量因子值
        """
        try:
            if not fundamentals:
                return FactorValue(
                    factor_name=self.name,
                    symbol=symbol,
                    value=0.0,
                    signal=0.0,
                )

            roe = fundamentals.get("roe", 0)
            debt_equity = fundamentals.get("debt_equity", 0)
            gross_margin = fundamentals.get("gross_margin", 0)
            earnings_growth = fundamentals.get("earnings_growth", 0)

            scores = []

            # ROE得分 (>15%为优秀)
            if roe is not None:
                roe_score = np.clip(roe / 15, 0, 1)
                scores.append(roe_score)

            # 负债率得分 (低负债好)
            if debt_equity is not None and debt_equity >= 0:
                de_score = np.clip(1 - debt_equity / 2, 0, 1)  # D/E>2为高风险
                scores.append(de_score)

            # 毛利率得分
            if gross_margin is not None and gross_margin > 0:
                gm_score = np.clip(gross_margin / 50, 0, 1)  # 50%毛利率为满分
                scores.append(gm_score)

            # 盈利增长得分
            if earnings_growth is not None:
                eg_score = np.clip((earnings_growth + 20) / 40, 0, 1)  # -20%到+20%
                scores.append(eg_score)

            if not scores:
                value = 0.0
                signal = 0.0
            else:
                value = np.mean(scores) * 100
                signal = np.clip((value - 50) / 50, -1, 1)

            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=value,
                signal=signal,
                metadata={
                    "roe": roe,
                    "debt_equity": debt_equity,
                    "gross_margin": gross_margin,
                    "earnings_growth": earnings_growth,
                },
            )

        except Exception as e:
            logger.error(f"Quality calculation failed for {symbol}: {e}")
            return FactorValue(
                factor_name=self.name,
                symbol=symbol,
                value=0.0,
                signal=0.0,
            )


class MultiFactorModel:
    """
    多因子组合模型

    组合多个因子模型，提供统一的因子评分。

    默认因子权重:
    - Momentum: 40%
    - Value: 30%
    - Quality: 30%
    """

    DEFAULT_WEIGHTS = {
        "momentum": 0.4,
        "value": 0.3,
        "quality": 0.3,
    }

    def __init__(
        self,
        factors: List[FactorModel] = None,
        weights: Dict[str, float] = None,
    ):
        """
        初始化多因子模型

        Args:
            factors: 因子模型列表
            weights: 因子权重字典
        """
        self.factors: Dict[str, FactorModel] = {}

        # 添加默认因子
        if factors is None:
            factors = [
                MomentumFactor(),
                ValueFactor(),
                QualityFactor(),
            ]

        for factor in factors:
            self.factors[factor.name] = factor

        # 设置权重
        self.weights = weights or self.DEFAULT_WEIGHTS

        # 确保权重归一化
        total_weight = sum(self.weights.get(f, 1.0) for f in self.factors)
        self.weights = {
            f: self.weights.get(f, 1.0) / total_weight
            for f in self.factors
        }

    def add_factor(self, factor: FactorModel, weight: float = None):
        """添加因子"""
        self.factors[factor.name] = factor
        if weight is not None:
            self.weights[factor.name] = weight
            # 重新归一化
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(
        self,
        data: pd.DataFrame,
        symbol: str,
        fundamentals: Dict[str, float] = None,
        **kwargs
    ) -> MultiFactorResult:
        """
        计算多因子综合得分

        Args:
            data: 价格数据
            symbol: 标的代码
            fundamentals: 基本面数据
            **kwargs: 传递给各因子的参数

        Returns:
            多因子结果
        """
        factor_values = []

        for name, factor in self.factors.items():
            if name == "value" or name == "quality":
                fv = factor.calculate(data, symbol, fundamentals=fundamentals, **kwargs)
            else:
                fv = factor.calculate(data, symbol, **kwargs)

            factor_values.append(fv)

        # 计算加权综合得分
        composite_score = 0.0
        composite_signal = 0.0
        total_weight = 0.0

        for fv in factor_values:
            weight = self.weights.get(fv.factor_name, 0)
            composite_score += fv.value * weight
            composite_signal += fv.signal * weight
            total_weight += weight

        if total_weight > 0:
            composite_signal = np.clip(composite_signal / total_weight, -1, 1)

        # 置信度基于因子覆盖度
        valid_factors = sum(1 for fv in factor_values if fv.value != 0)
        confidence = valid_factors / len(self.factors)

        return MultiFactorResult(
            symbol=symbol,
            composite_score=composite_score,
            composite_signal=composite_signal,
            factor_values=factor_values,
            factor_weights=self.weights.copy(),
            confidence=confidence,
        )

    def rank_symbols(
        self,
        symbols_data: Dict[str, Tuple[pd.DataFrame, Dict[str, float]]],
    ) -> List[Tuple[str, float]]:
        """
        对多个标的进行因子排名

        Args:
            symbols_data: {symbol: (price_data, fundamentals)}

        Returns:
            排序后的(symbol, score)列表
        """
        results = []

        for symbol, (data, fundamentals) in symbols_data.items():
            result = self.calculate(data, symbol, fundamentals)
            results.append((symbol, result.composite_score))

        # 按得分降序排列
        results.sort(key=lambda x: x[1], reverse=True)

        return results


# ============ 便捷函数 ============

_factor_models: Dict[str, FactorModel] = {}
_multi_factor_model: Optional[MultiFactorModel] = None


def get_factor_model(factor_type: str = "momentum", **kwargs) -> FactorModel:
    """
    获取或创建因子模型

    Args:
        factor_type: 因子类型 (momentum, value, quality)
        **kwargs: 因子参数

    Returns:
        FactorModel实例
    """
    if factor_type not in _factor_models:
        if factor_type == "momentum":
            _factor_models[factor_type] = MomentumFactor(**kwargs)
        elif factor_type == "value":
            _factor_models[factor_type] = ValueFactor(**kwargs)
        elif factor_type == "quality":
            _factor_models[factor_type] = QualityFactor(**kwargs)
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")

    return _factor_models[factor_type]


def get_multi_factor_model(
    weights: Dict[str, float] = None,
) -> MultiFactorModel:
    """获取多因子模型单例"""
    global _multi_factor_model
    if _multi_factor_model is None:
        _multi_factor_model = MultiFactorModel(weights=weights)
    return _multi_factor_model
