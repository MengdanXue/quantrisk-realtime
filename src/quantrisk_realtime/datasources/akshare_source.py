"""
Akshare A股数据源 (Akshare A-Share Data Source)

提供中国A股市场的数据访问:
- 股票日线/分钟线数据
- 实时行情
- 指数数据
- 板块数据
- 龙虎榜数据
- 融资融券数据

完全免费，无API限制。

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

import pandas as pd

from .base import (
    DataSource,
    DataSourceConfig,
    DataSourceState,
    DataSourceError,
    DataNotFoundError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
)

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("akshare未安装，请运行: pip install akshare")


# ==================== 枚举类型 ====================

class AkshareMarket(str, Enum):
    """A股市场类型"""
    SH = "sh"       # 上海
    SZ = "sz"       # 深圳
    BJ = "bj"       # 北京
    HK = "hk"       # 港股通


class AkshareIndexType(str, Enum):
    """指数类型"""
    MAIN = "main"           # 主要指数
    INDUSTRY = "industry"   # 行业指数
    CONCEPT = "concept"     # 概念指数
    STYLE = "style"         # 风格指数


# ==================== 配置类 ====================

@dataclass
class AkshareConfig(DataSourceConfig):
    """Akshare配置"""
    name: str = "akshare"
    rate_limit_per_second: float = 5.0
    rate_limit_per_minute: float = 100.0
    cache_ttl: float = 30.0

    # A股特定配置
    adjust: str = "qfq"  # 复权类型: qfq前复权, hfq后复权, None不复权
    default_period: str = "daily"  # daily, weekly, monthly

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'adjust': self.adjust,
            'default_period': self.default_period,
        })
        return base


# ==================== 数据类 ====================

@dataclass
class StockInfo:
    """股票信息"""
    symbol: str
    name: str
    market: AkshareMarket
    industry: str = ""
    list_date: Optional[datetime] = None
    total_shares: float = 0.0
    circulating_shares: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'market': self.market.value,
            'industry': self.industry,
            'list_date': self.list_date.isoformat() if self.list_date else None,
            'total_shares': self.total_shares,
            'circulating_shares': self.circulating_shares,
        }


@dataclass
class MarginData:
    """融资融券数据"""
    date: datetime
    symbol: str
    margin_balance: float  # 融资余额
    margin_buy: float      # 融资买入
    short_balance: float   # 融券余额
    short_sell: float      # 融券卖出

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'symbol': self.symbol,
            'margin_balance': self.margin_balance,
            'margin_buy': self.margin_buy,
            'short_balance': self.short_balance,
            'short_sell': self.short_sell,
        }


@dataclass
class TopListData:
    """龙虎榜数据"""
    date: datetime
    symbol: str
    name: str
    close: float
    change_pct: float
    turnover_rate: float
    net_buy: float  # 净买入
    buy_amount: float
    sell_amount: float
    reason: str  # 上榜原因

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'symbol': self.symbol,
            'name': self.name,
            'close': self.close,
            'change_pct': self.change_pct,
            'turnover_rate': self.turnover_rate,
            'net_buy': self.net_buy,
            'buy_amount': self.buy_amount,
            'sell_amount': self.sell_amount,
            'reason': self.reason,
        }


# ==================== 主类 ====================

class AkshareDataSource(DataSource):
    """Akshare A股数据源"""

    def __init__(self, config: Optional[AkshareConfig] = None):
        super().__init__(config or AkshareConfig())
        self._stock_list_cache: Optional[pd.DataFrame] = None
        self._stock_list_update_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "akshare"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.STOCK_CN, MarketType.INDEX]

    async def connect(self) -> bool:
        """连接数据源"""
        if not AKSHARE_AVAILABLE:
            logger.error("akshare库未安装")
            self.state = DataSourceState.ERROR
            return False

        self.state = DataSourceState.RUNNING
        logger.info("Akshare数据源已连接")
        return True

    async def disconnect(self) -> None:
        """断开连接"""
        self.state = DataSourceState.IDLE
        logger.info("Akshare数据源已断开")

    # ==================== OHLCV数据 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取A股OHLCV数据"""
        cache_key = f"ohlcv:{symbol}:{frequency}:{start_date}:{end_date}:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_ohlcv_impl,
                symbol, frequency, start_date, end_date, limit
            )
            await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取{symbol} OHLCV失败: {e}")
            raise DataSourceError(f"Failed to fetch OHLCV for {symbol}: {e}")

    async def _fetch_ohlcv_impl(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int,
    ) -> List[OHLCV]:
        """OHLCV实现"""
        # 标准化symbol
        clean_symbol = self._normalize_symbol(symbol)

        # 设置日期范围
        end_date = end_date or datetime.now()
        if start_date is None:
            if frequency == DataFrequency.DAILY:
                start_date = end_date - timedelta(days=limit * 2)
            else:
                start_date = end_date - timedelta(days=30)

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # 获取数据
        loop = asyncio.get_event_loop()

        if frequency == DataFrequency.DAILY:
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_zh_a_hist(
                    symbol=clean_symbol,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust=self.config.adjust
                )
            )
        elif frequency in [DataFrequency.MINUTE_1, DataFrequency.MINUTE_5,
                          DataFrequency.MINUTE_15, DataFrequency.MINUTE_30,
                          DataFrequency.HOUR_1]:
            period_map = {
                DataFrequency.MINUTE_1: "1",
                DataFrequency.MINUTE_5: "5",
                DataFrequency.MINUTE_15: "15",
                DataFrequency.MINUTE_30: "30",
                DataFrequency.HOUR_1: "60",
            }
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_zh_a_hist_min_em(
                    symbol=clean_symbol,
                    period=period_map[frequency],
                    adjust=self.config.adjust
                )
            )
        else:
            raise DataSourceError(f"Unsupported frequency: {frequency}")

        if df is None or df.empty:
            return []

        # 转换为OHLCV列表
        results = []
        for _, row in df.iterrows():
            try:
                ohlcv = OHLCV(
                    timestamp=pd.to_datetime(row.get('日期') or row.get('时间')),
                    open=float(row['开盘']),
                    high=float(row['最高']),
                    low=float(row['最低']),
                    close=float(row['收盘']),
                    volume=float(row['成交量']),
                    symbol=symbol,
                    source=self.name,
                )
                results.append(ohlcv)
            except Exception as e:
                logger.warning(f"解析行数据失败: {e}")
                continue

        return results[-limit:] if len(results) > limit else results

    # ==================== 实时报价 ====================

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取A股实时报价"""
        cache_key = f"quote:{symbol}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_quote_impl, symbol
            )
            if result:
                await self._set_cached(cache_key, result, ttl=5.0)  # 报价缓存5秒
            return result
        except Exception as e:
            logger.error(f"获取{symbol}报价失败: {e}")
            return None

    async def _fetch_quote_impl(self, symbol: str) -> Optional[Quote]:
        """报价实现"""
        clean_symbol = self._normalize_symbol(symbol)

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_zh_a_spot_em()
        )

        if df is None or df.empty:
            return None

        # 查找symbol
        row = df[df['代码'] == clean_symbol]
        if row.empty:
            return None

        row = row.iloc[0]

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=float(row.get('买入', 0) or 0),
            ask=float(row.get('卖出', 0) or 0),
            last_price=float(row.get('最新价', 0) or 0),
            volume=float(row.get('成交量', 0) or 0),
            source=self.name,
        )

    # ==================== 股票列表 ====================

    async def get_stock_list(self, market: Optional[AkshareMarket] = None) -> List[StockInfo]:
        """获取股票列表"""
        # 检查缓存
        if (self._stock_list_cache is not None and
            self._stock_list_update_time and
            (datetime.now() - self._stock_list_update_time).seconds < 3600):
            df = self._stock_list_cache
        else:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_zh_a_spot_em()
            )
            self._stock_list_cache = df
            self._stock_list_update_time = datetime.now()

        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            code = str(row['代码'])

            # 判断市场
            if code.startswith('6'):
                stock_market = AkshareMarket.SH
            elif code.startswith(('0', '3')):
                stock_market = AkshareMarket.SZ
            elif code.startswith(('4', '8')):
                stock_market = AkshareMarket.BJ
            else:
                continue

            # 过滤市场
            if market and stock_market != market:
                continue

            results.append(StockInfo(
                symbol=code,
                name=str(row['名称']),
                market=stock_market,
            ))

        return results

    # ==================== 指数数据 ====================

    async def fetch_index_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取指数OHLCV数据"""
        cache_key = f"index_ohlcv:{symbol}:{start_date}:{end_date}:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_index_ohlcv_impl,
                symbol, start_date, end_date, limit
            )
            await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取指数{symbol} OHLCV失败: {e}")
            raise DataSourceError(f"Failed to fetch index OHLCV for {symbol}: {e}")

    async def _fetch_index_ohlcv_impl(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int,
    ) -> List[OHLCV]:
        """指数OHLCV实现"""
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=limit * 2))

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        )

        if df is None or df.empty:
            # 尝试深证指数
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_zh_index_daily(symbol=f"sz{symbol}")
            )

        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            try:
                ohlcv = OHLCV(
                    timestamp=pd.to_datetime(row['date']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    symbol=symbol,
                    source=self.name,
                )
                results.append(ohlcv)
            except Exception as e:
                continue

        return results[-limit:] if len(results) > limit else results

    # ==================== 融资融券数据 ====================

    async def fetch_margin_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MarginData]:
        """获取融资融券数据"""
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_margin_detail_szse(symbol=self._normalize_symbol(symbol))
            )

            if df is None or df.empty:
                return []

            results = []
            for _, row in df.iterrows():
                try:
                    results.append(MarginData(
                        date=pd.to_datetime(row['信用交易日期']),
                        symbol=symbol,
                        margin_balance=float(row.get('融资余额', 0) or 0),
                        margin_buy=float(row.get('融资买入额', 0) or 0),
                        short_balance=float(row.get('融券余额', 0) or 0),
                        short_sell=float(row.get('融券卖出量', 0) or 0),
                    ))
                except Exception:
                    continue

            return results
        except Exception as e:
            logger.error(f"获取融资融券数据失败: {e}")
            return []

    # ==================== 龙虎榜数据 ====================

    async def fetch_top_list(
        self,
        date: Optional[datetime] = None,
    ) -> List[TopListData]:
        """获取龙虎榜数据"""
        date = date or datetime.now()
        date_str = date.strftime("%Y%m%d")

        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ak.stock_lhb_detail_em(date=date_str)
            )

            if df is None or df.empty:
                return []

            results = []
            for _, row in df.iterrows():
                try:
                    results.append(TopListData(
                        date=date,
                        symbol=str(row['代码']),
                        name=str(row['名称']),
                        close=float(row.get('收盘价', 0) or 0),
                        change_pct=float(row.get('涨跌幅', 0) or 0),
                        turnover_rate=float(row.get('换手率', 0) or 0),
                        net_buy=float(row.get('净买额', 0) or 0),
                        buy_amount=float(row.get('买入额', 0) or 0),
                        sell_amount=float(row.get('卖出额', 0) or 0),
                        reason=str(row.get('上榜原因', '')),
                    ))
                except Exception:
                    continue

            return results
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败: {e}")
            return []

    # ==================== 辅助方法 ====================

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化股票代码"""
        # 移除前缀
        symbol = symbol.upper()
        for prefix in ['SH', 'SZ', 'BJ', '.SS', '.SZ']:
            symbol = symbol.replace(prefix, '')
        return symbol.strip()


# ==================== 工厂函数 ====================

def create_akshare_source(
    adjust: str = "qfq",
    cache_ttl: float = 30.0,
) -> AkshareDataSource:
    """创建Akshare数据源"""
    config = AkshareConfig(
        adjust=adjust,
        cache_ttl=cache_ttl,
    )
    return AkshareDataSource(config)
