"""
数据源模块 (Data Sources Module)

提供多种免费和低成本金融数据源的统一接口:
- Akshare: A股数据
- CCXT: 加密货币统一接口
- CoinGecko: 加密货币市场数据
- FRED: 美联储经济数据
- Finnhub: 美股实时数据
- AlternativeMe: 恐惧贪婪指数

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from .akshare_source import (
    AkshareDataSource,
    AkshareConfig,
    create_akshare_source,
)

from .ccxt_source import (
    CCXTDataSource,
    CCXTConfig,
    ExchangeType,
    create_ccxt_source,
)

from .coingecko_source import (
    CoinGeckoDataSource,
    CoinGeckoConfig,
    create_coingecko_source,
)

from .fred_source import (
    FREDDataSource,
    FREDConfig,
    EconomicIndicator,
    create_fred_source,
)

from .fear_greed_source import (
    FearGreedDataSource,
    FearGreedConfig,
    SentimentLevel,
    create_fear_greed_source,
)

from .finnhub_source import (
    FinnhubDataSource,
    FinnhubConfig,
    CompanyProfile,
    BasicFinancials,
    CompanyNews,
    EarningsCalendar,
    RecommendationTrend,
    PriceTarget,
    create_finnhub_source,
)

from .manager import (
    DataSourceManager,
    ManagerConfig,
    SelectionStrategy,
    AggregationMethod,
    HealthStatus,
    AggregatedQuote,
    create_manager,
    create_default_manager,
)

from .base import (
    DataSource,
    DataSourceConfig,
    DataSourceError,
    RateLimitError,
    DataNotFoundError,
)

__all__ = [
    # Base
    'DataSource',
    'DataSourceConfig',
    'DataSourceError',
    'RateLimitError',
    'DataNotFoundError',
    # Manager
    'DataSourceManager',
    'ManagerConfig',
    'SelectionStrategy',
    'AggregationMethod',
    'HealthStatus',
    'AggregatedQuote',
    'create_manager',
    'create_default_manager',
    # Akshare
    'AkshareDataSource',
    'AkshareConfig',
    'create_akshare_source',
    # CCXT
    'CCXTDataSource',
    'CCXTConfig',
    'ExchangeType',
    'create_ccxt_source',
    # CoinGecko
    'CoinGeckoDataSource',
    'CoinGeckoConfig',
    'create_coingecko_source',
    # FRED
    'FREDDataSource',
    'FREDConfig',
    'EconomicIndicator',
    'create_fred_source',
    # Fear & Greed
    'FearGreedDataSource',
    'FearGreedConfig',
    'SentimentLevel',
    'create_fear_greed_source',
    # Finnhub
    'FinnhubDataSource',
    'FinnhubConfig',
    'CompanyProfile',
    'BasicFinancials',
    'CompanyNews',
    'EarningsCalendar',
    'RecommendationTrend',
    'PriceTarget',
    'create_finnhub_source',
]
