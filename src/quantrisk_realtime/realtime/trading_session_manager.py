"""
交易会话管理器 (Trading Session Manager)

提供全球市场交易时段管理、交易日历、节假日处理和订单时段控制。

主要功能:
- 交易时段定义与管理
- 全球市场交易日历
- 节假日管理
- 订单时段验证
- 实时市场状态监控
- 跨时区支持

支持市场:
- 美股 (NYSE, NASDAQ)
- 港股 (HKEX)
- A股 (SSE, SZSE)
- 期货 (CME, NYMEX, ICE)
- 外汇 (24小时)
- 加密货币 (24/7)
"""

import asyncio
from datetime import datetime, date, time, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from zoneinfo import ZoneInfo
import threading
import logging

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class MarketType(str, Enum):
    """市场类型"""
    STOCK = "stock"
    FUTURES = "futures"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTIONS = "options"
    BOND = "bond"
    COMMODITY = "commodity"


class SessionType(str, Enum):
    """交易时段类型"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    OVERNIGHT = "overnight"
    AUCTION_OPEN = "auction_open"
    AUCTION_CLOSE = "auction_close"
    LUNCH_BREAK = "lunch_break"
    CLOSED = "closed"


class MarketStatus(str, Enum):
    """市场状态"""
    OPEN = "open"
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    POST_CLOSE = "post_close"
    HALTED = "halted"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"
    EARLY_CLOSE = "early_close"


class HolidayType(str, Enum):
    """节假日类型"""
    FULL_DAY = "full_day"
    EARLY_CLOSE = "early_close"
    LATE_OPEN = "late_open"


class OrderSessionRule(str, Enum):
    """订单时段规则"""
    REGULAR_ONLY = "regular_only"
    EXTENDED_HOURS = "extended_hours"
    ANY_SESSION = "any_session"
    MARKET_HOURS_ONLY = "market_hours_only"


# ==================== 数据类定义 ====================

@dataclass
class TradingSession:
    """交易时段"""
    session_type: SessionType
    start_time: time
    end_time: time
    timezone: str = "America/New_York"
    allow_orders: bool = True
    allow_market_orders: bool = True
    allow_limit_orders: bool = True
    description: str = ""

    def contains_time(self, t: time) -> bool:
        """检查时间是否在时段内"""
        if self.start_time <= self.end_time:
            return self.start_time <= t <= self.end_time
        else:
            # 跨午夜时段
            return t >= self.start_time or t <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_type": self.session_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "timezone": self.timezone,
            "allow_orders": self.allow_orders,
            "allow_market_orders": self.allow_market_orders,
            "allow_limit_orders": self.allow_limit_orders,
            "description": self.description,
        }


@dataclass
class Holiday:
    """节假日"""
    date: date
    name: str
    holiday_type: HolidayType = HolidayType.FULL_DAY
    early_close_time: Optional[time] = None
    late_open_time: Optional[time] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "date": self.date.isoformat(),
            "name": self.name,
            "holiday_type": self.holiday_type.value,
            "early_close_time": self.early_close_time.isoformat() if self.early_close_time else None,
            "late_open_time": self.late_open_time.isoformat() if self.late_open_time else None,
            "description": self.description,
        }


@dataclass
class MarketCalendar:
    """市场日历"""
    market_code: str
    market_name: str
    market_type: MarketType
    timezone: str
    sessions: List[TradingSession] = field(default_factory=list)
    holidays: Dict[date, Holiday] = field(default_factory=dict)
    trading_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # 周一到周五
    currency: str = "USD"

    def is_trading_day(self, d: date) -> bool:
        """检查是否为交易日"""
        # 检查是否为节假日
        if d in self.holidays:
            holiday = self.holidays[d]
            if holiday.holiday_type == HolidayType.FULL_DAY:
                return False

        # 检查是否为交易日（周几）
        return d.weekday() in self.trading_days

    def get_holiday(self, d: date) -> Optional[Holiday]:
        """获取节假日信息"""
        return self.holidays.get(d)

    def get_current_session(self, dt: Optional[datetime] = None) -> Optional[TradingSession]:
        """获取当前交易时段"""
        if dt is None:
            dt = datetime.now(ZoneInfo(self.timezone))
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(self.timezone))

        current_time = dt.time()

        for session in self.sessions:
            if session.contains_time(current_time):
                return session

        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "market_code": self.market_code,
            "market_name": self.market_name,
            "market_type": self.market_type.value,
            "timezone": self.timezone,
            "sessions": [s.to_dict() for s in self.sessions],
            "trading_days": list(self.trading_days),
            "currency": self.currency,
        }


@dataclass
class SessionStatus:
    """会话状态"""
    market_code: str
    market_status: MarketStatus
    current_session: Optional[SessionType]
    next_session: Optional[SessionType]
    next_session_time: Optional[datetime]
    is_trading_day: bool
    holiday: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "market_code": self.market_code,
            "market_status": self.market_status.value,
            "current_session": self.current_session.value if self.current_session else None,
            "next_session": self.next_session.value if self.next_session else None,
            "next_session_time": self.next_session_time.isoformat() if self.next_session_time else None,
            "is_trading_day": self.is_trading_day,
            "holiday": self.holiday,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderValidation:
    """订单验证结果"""
    is_valid: bool
    market_code: str
    session_type: Optional[SessionType]
    reason: str = ""
    allows_market_orders: bool = False
    allows_limit_orders: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "market_code": self.market_code,
            "session_type": self.session_type.value if self.session_type else None,
            "reason": self.reason,
            "allows_market_orders": self.allows_market_orders,
            "allows_limit_orders": self.allows_limit_orders,
        }


@dataclass
class ManagerConfig:
    """管理器配置"""
    auto_update: bool = True
    update_interval: float = 60.0  # 秒
    cache_holidays: bool = True
    enable_notifications: bool = True


# ==================== 预定义市场日历 ====================

def create_us_stock_calendar() -> MarketCalendar:
    """创建美股日历"""
    sessions = [
        TradingSession(
            session_type=SessionType.PRE_MARKET,
            start_time=time(4, 0),
            end_time=time(9, 30),
            timezone="America/New_York",
            allow_orders=True,
            allow_market_orders=False,
            allow_limit_orders=True,
            description="盘前交易",
        ),
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(16, 0),
            timezone="America/New_York",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="常规交易时段",
        ),
        TradingSession(
            session_type=SessionType.POST_MARKET,
            start_time=time(16, 0),
            end_time=time(20, 0),
            timezone="America/New_York",
            allow_orders=True,
            allow_market_orders=False,
            allow_limit_orders=True,
            description="盘后交易",
        ),
    ]

    # 2024-2025年美国股市节假日
    holidays = {
        date(2024, 1, 1): Holiday(date(2024, 1, 1), "New Year's Day"),
        date(2024, 1, 15): Holiday(date(2024, 1, 15), "Martin Luther King Jr. Day"),
        date(2024, 2, 19): Holiday(date(2024, 2, 19), "Presidents' Day"),
        date(2024, 3, 29): Holiday(date(2024, 3, 29), "Good Friday"),
        date(2024, 5, 27): Holiday(date(2024, 5, 27), "Memorial Day"),
        date(2024, 6, 19): Holiday(date(2024, 6, 19), "Juneteenth"),
        date(2024, 7, 4): Holiday(date(2024, 7, 4), "Independence Day"),
        date(2024, 9, 2): Holiday(date(2024, 9, 2), "Labor Day"),
        date(2024, 11, 28): Holiday(date(2024, 11, 28), "Thanksgiving Day"),
        date(2024, 11, 29): Holiday(date(2024, 11, 29), "Day After Thanksgiving", HolidayType.EARLY_CLOSE, time(13, 0)),
        date(2024, 12, 25): Holiday(date(2024, 12, 25), "Christmas Day"),
        date(2025, 1, 1): Holiday(date(2025, 1, 1), "New Year's Day"),
        date(2025, 1, 20): Holiday(date(2025, 1, 20), "Martin Luther King Jr. Day"),
        date(2025, 2, 17): Holiday(date(2025, 2, 17), "Presidents' Day"),
        date(2025, 4, 18): Holiday(date(2025, 4, 18), "Good Friday"),
        date(2025, 5, 26): Holiday(date(2025, 5, 26), "Memorial Day"),
        date(2025, 6, 19): Holiday(date(2025, 6, 19), "Juneteenth"),
        date(2025, 7, 4): Holiday(date(2025, 7, 4), "Independence Day"),
        date(2025, 9, 1): Holiday(date(2025, 9, 1), "Labor Day"),
        date(2025, 11, 27): Holiday(date(2025, 11, 27), "Thanksgiving Day"),
        date(2025, 12, 25): Holiday(date(2025, 12, 25), "Christmas Day"),
    }

    return MarketCalendar(
        market_code="US",
        market_name="US Stock Market",
        market_type=MarketType.STOCK,
        timezone="America/New_York",
        sessions=sessions,
        holidays=holidays,
        trading_days={0, 1, 2, 3, 4},
        currency="USD",
    )


def create_hk_stock_calendar() -> MarketCalendar:
    """创建港股日历"""
    sessions = [
        TradingSession(
            session_type=SessionType.AUCTION_OPEN,
            start_time=time(9, 0),
            end_time=time(9, 30),
            timezone="Asia/Hong_Kong",
            allow_orders=True,
            allow_market_orders=False,
            allow_limit_orders=True,
            description="开盘竞价",
        ),
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(12, 0),
            timezone="Asia/Hong_Kong",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="上午交易时段",
        ),
        TradingSession(
            session_type=SessionType.LUNCH_BREAK,
            start_time=time(12, 0),
            end_time=time(13, 0),
            timezone="Asia/Hong_Kong",
            allow_orders=False,
            description="午休",
        ),
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(13, 0),
            end_time=time(16, 0),
            timezone="Asia/Hong_Kong",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="下午交易时段",
        ),
        TradingSession(
            session_type=SessionType.AUCTION_CLOSE,
            start_time=time(16, 0),
            end_time=time(16, 10),
            timezone="Asia/Hong_Kong",
            allow_orders=True,
            allow_market_orders=False,
            allow_limit_orders=True,
            description="收盘竞价",
        ),
    ]

    return MarketCalendar(
        market_code="HK",
        market_name="Hong Kong Stock Exchange",
        market_type=MarketType.STOCK,
        timezone="Asia/Hong_Kong",
        sessions=sessions,
        holidays={},
        trading_days={0, 1, 2, 3, 4},
        currency="HKD",
    )


def create_cn_stock_calendar() -> MarketCalendar:
    """创建A股日历"""
    sessions = [
        TradingSession(
            session_type=SessionType.AUCTION_OPEN,
            start_time=time(9, 15),
            end_time=time(9, 25),
            timezone="Asia/Shanghai",
            allow_orders=True,
            allow_market_orders=False,
            allow_limit_orders=True,
            description="集合竞价",
        ),
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(11, 30),
            timezone="Asia/Shanghai",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="上午交易时段",
        ),
        TradingSession(
            session_type=SessionType.LUNCH_BREAK,
            start_time=time(11, 30),
            end_time=time(13, 0),
            timezone="Asia/Shanghai",
            allow_orders=False,
            description="午休",
        ),
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(13, 0),
            end_time=time(15, 0),
            timezone="Asia/Shanghai",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="下午交易时段",
        ),
    ]

    return MarketCalendar(
        market_code="CN",
        market_name="China A-Share Market",
        market_type=MarketType.STOCK,
        timezone="Asia/Shanghai",
        sessions=sessions,
        holidays={},
        trading_days={0, 1, 2, 3, 4},
        currency="CNY",
    )


def create_futures_calendar() -> MarketCalendar:
    """创建期货日历（CME）"""
    sessions = [
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(18, 0),
            end_time=time(17, 0),  # 跨午夜
            timezone="America/Chicago",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="电子交易时段（几乎24小时）",
        ),
    ]

    return MarketCalendar(
        market_code="CME",
        market_name="Chicago Mercantile Exchange",
        market_type=MarketType.FUTURES,
        timezone="America/Chicago",
        sessions=sessions,
        holidays={},
        trading_days={0, 1, 2, 3, 4, 6},  # 周一到周五+周日晚
        currency="USD",
    )


def create_forex_calendar() -> MarketCalendar:
    """创建外汇日历"""
    sessions = [
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(17, 0),  # 周日下午5点开始
            end_time=time(17, 0),    # 周五下午5点结束
            timezone="America/New_York",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="24小时交易（周一至周五）",
        ),
    ]

    return MarketCalendar(
        market_code="FX",
        market_name="Foreign Exchange Market",
        market_type=MarketType.FOREX,
        timezone="America/New_York",
        sessions=sessions,
        holidays={},
        trading_days={0, 1, 2, 3, 4},
        currency="USD",
    )


def create_crypto_calendar() -> MarketCalendar:
    """创建加密货币日历"""
    sessions = [
        TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(0, 0),
            end_time=time(23, 59, 59),
            timezone="UTC",
            allow_orders=True,
            allow_market_orders=True,
            allow_limit_orders=True,
            description="24/7交易",
        ),
    ]

    return MarketCalendar(
        market_code="CRYPTO",
        market_name="Cryptocurrency Market",
        market_type=MarketType.CRYPTO,
        timezone="UTC",
        sessions=sessions,
        holidays={},
        trading_days={0, 1, 2, 3, 4, 5, 6},  # 每天
        currency="USD",
    )


# ==================== 交易会话管理器 ====================

class TradingSessionManager:
    """交易会话管理器"""

    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()

        # 市场日历
        self.calendars: Dict[str, MarketCalendar] = {}

        # 状态缓存
        self._status_cache: Dict[str, SessionStatus] = {}
        self._cache_time: Optional[datetime] = None

        # 回调
        self._session_callbacks: List[Callable[[str, SessionType, SessionType], None]] = []
        self._status_callbacks: List[Callable[[str, MarketStatus], None]] = []

        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.Lock()

        # 初始化预定义市场
        self._init_default_markets()

    def _init_default_markets(self):
        """初始化默认市场"""
        self.register_market(create_us_stock_calendar())
        self.register_market(create_hk_stock_calendar())
        self.register_market(create_cn_stock_calendar())
        self.register_market(create_futures_calendar())
        self.register_market(create_forex_calendar())
        self.register_market(create_crypto_calendar())

    def register_market(self, calendar: MarketCalendar):
        """注册市场日历"""
        self.calendars[calendar.market_code] = calendar
        logger.info(f"Registered market: {calendar.market_code} - {calendar.market_name}")

    def unregister_market(self, market_code: str) -> bool:
        """注销市场日历"""
        if market_code in self.calendars:
            del self.calendars[market_code]
            return True
        return False

    def get_calendar(self, market_code: str) -> Optional[MarketCalendar]:
        """获取市场日历"""
        return self.calendars.get(market_code)

    def get_all_markets(self) -> List[str]:
        """获取所有市场代码"""
        return list(self.calendars.keys())

    def get_market_status(self, market_code: str, dt: Optional[datetime] = None) -> SessionStatus:
        """获取市场状态"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return SessionStatus(
                market_code=market_code,
                market_status=MarketStatus.CLOSED,
                current_session=None,
                next_session=None,
                next_session_time=None,
                is_trading_day=False,
                holiday=None,
            )

        if dt is None:
            dt = datetime.now(ZoneInfo(calendar.timezone))
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(calendar.timezone))

        current_date = dt.date()

        # 检查是否为交易日
        is_trading_day = calendar.is_trading_day(current_date)
        holiday = calendar.get_holiday(current_date)
        holiday_name = holiday.name if holiday else None

        # 检查周末
        if current_date.weekday() not in calendar.trading_days:
            return SessionStatus(
                market_code=market_code,
                market_status=MarketStatus.WEEKEND,
                current_session=None,
                next_session=SessionType.REGULAR,
                next_session_time=self._get_next_trading_day_open(calendar, current_date),
                is_trading_day=False,
                holiday=None,
            )

        # 检查节假日
        if holiday and holiday.holiday_type == HolidayType.FULL_DAY:
            return SessionStatus(
                market_code=market_code,
                market_status=MarketStatus.HOLIDAY,
                current_session=None,
                next_session=SessionType.REGULAR,
                next_session_time=self._get_next_trading_day_open(calendar, current_date),
                is_trading_day=False,
                holiday=holiday_name,
            )

        # 获取当前时段
        current_session = calendar.get_current_session(dt)

        if current_session:
            status = MarketStatus.OPEN
            if current_session.session_type == SessionType.PRE_MARKET:
                status = MarketStatus.PRE_OPEN
            elif current_session.session_type == SessionType.POST_MARKET:
                status = MarketStatus.POST_CLOSE
            elif current_session.session_type == SessionType.LUNCH_BREAK:
                status = MarketStatus.CLOSED

            return SessionStatus(
                market_code=market_code,
                market_status=status,
                current_session=current_session.session_type,
                next_session=self._get_next_session_type(calendar, current_session),
                next_session_time=self._get_next_session_time(calendar, dt, current_session),
                is_trading_day=is_trading_day,
                holiday=holiday_name,
            )

        # 市场休市
        return SessionStatus(
            market_code=market_code,
            market_status=MarketStatus.CLOSED,
            current_session=None,
            next_session=SessionType.REGULAR,
            next_session_time=self._get_next_trading_day_open(calendar, current_date),
            is_trading_day=is_trading_day,
            holiday=holiday_name,
        )

    def _get_next_session_type(self, calendar: MarketCalendar, current: TradingSession) -> Optional[SessionType]:
        """获取下一个时段类型"""
        sessions = calendar.sessions
        current_idx = next((i for i, s in enumerate(sessions) if s.session_type == current.session_type), -1)

        if current_idx >= 0 and current_idx < len(sessions) - 1:
            return sessions[current_idx + 1].session_type
        elif current_idx == len(sessions) - 1:
            return sessions[0].session_type if sessions else None

        return None

    def _get_next_session_time(
        self, calendar: MarketCalendar, dt: datetime, current: TradingSession
    ) -> Optional[datetime]:
        """获取下一个时段开始时间"""
        sessions = calendar.sessions
        current_idx = next((i for i, s in enumerate(sessions) if s.session_type == current.session_type), -1)

        if current_idx >= 0 and current_idx < len(sessions) - 1:
            next_session = sessions[current_idx + 1]
            return dt.replace(
                hour=next_session.start_time.hour,
                minute=next_session.start_time.minute,
                second=0,
                microsecond=0,
            )

        # 下一个交易日
        return self._get_next_trading_day_open(calendar, dt.date())

    def _get_next_trading_day_open(self, calendar: MarketCalendar, d: date) -> Optional[datetime]:
        """获取下一个交易日开盘时间"""
        next_day = d + timedelta(days=1)

        # 最多查找30天
        for _ in range(30):
            if calendar.is_trading_day(next_day):
                # 找到第一个交易时段
                for session in calendar.sessions:
                    if session.session_type in (SessionType.PRE_MARKET, SessionType.REGULAR, SessionType.AUCTION_OPEN):
                        return datetime.combine(
                            next_day,
                            session.start_time,
                            tzinfo=ZoneInfo(calendar.timezone),
                        )
            next_day += timedelta(days=1)

        return None

    def validate_order(
        self,
        market_code: str,
        order_type: str = "limit",
        session_rule: OrderSessionRule = OrderSessionRule.REGULAR_ONLY,
        dt: Optional[datetime] = None,
    ) -> OrderValidation:
        """验证订单是否可以在当前时段提交"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return OrderValidation(
                is_valid=False,
                market_code=market_code,
                session_type=None,
                reason=f"Unknown market: {market_code}",
            )

        status = self.get_market_status(market_code, dt)

        # 检查是否为交易日
        if not status.is_trading_day:
            return OrderValidation(
                is_valid=False,
                market_code=market_code,
                session_type=None,
                reason=f"Not a trading day: {status.holiday or 'Weekend'}",
            )

        # 检查市场状态
        if status.market_status in (MarketStatus.HOLIDAY, MarketStatus.WEEKEND, MarketStatus.HALTED):
            return OrderValidation(
                is_valid=False,
                market_code=market_code,
                session_type=status.current_session,
                reason=f"Market is {status.market_status.value}",
            )

        # 获取当前时段
        current_session = None
        for session in calendar.sessions:
            if session.session_type == status.current_session:
                current_session = session
                break

        if not current_session:
            return OrderValidation(
                is_valid=False,
                market_code=market_code,
                session_type=None,
                reason="Market is closed",
            )

        # 检查时段规则
        if session_rule == OrderSessionRule.REGULAR_ONLY:
            if current_session.session_type != SessionType.REGULAR:
                return OrderValidation(
                    is_valid=False,
                    market_code=market_code,
                    session_type=current_session.session_type,
                    reason=f"Order only allowed during regular hours, current: {current_session.session_type.value}",
                )

        # 检查订单类型
        is_market_order = order_type.lower() == "market"
        if is_market_order and not current_session.allow_market_orders:
            return OrderValidation(
                is_valid=False,
                market_code=market_code,
                session_type=current_session.session_type,
                reason="Market orders not allowed in this session",
                allows_limit_orders=current_session.allow_limit_orders,
            )

        return OrderValidation(
            is_valid=True,
            market_code=market_code,
            session_type=current_session.session_type,
            reason="Order validated",
            allows_market_orders=current_session.allow_market_orders,
            allows_limit_orders=current_session.allow_limit_orders,
        )

    def is_market_open(self, market_code: str, dt: Optional[datetime] = None) -> bool:
        """检查市场是否开放"""
        status = self.get_market_status(market_code, dt)
        return status.market_status in (MarketStatus.OPEN, MarketStatus.PRE_OPEN, MarketStatus.POST_CLOSE)

    def is_regular_hours(self, market_code: str, dt: Optional[datetime] = None) -> bool:
        """检查是否为常规交易时段"""
        status = self.get_market_status(market_code, dt)
        return status.current_session == SessionType.REGULAR

    def get_trading_days(
        self,
        market_code: str,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """获取指定日期范围内的交易日"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return []

        trading_days = []
        current = start_date
        while current <= end_date:
            if calendar.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def get_holidays(
        self,
        market_code: str,
        year: Optional[int] = None,
    ) -> List[Holiday]:
        """获取节假日列表"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return []

        if year:
            return [h for h in calendar.holidays.values() if h.date.year == year]

        return list(calendar.holidays.values())

    def add_holiday(self, market_code: str, holiday: Holiday) -> bool:
        """添加节假日"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return False

        calendar.holidays[holiday.date] = holiday
        return True

    def remove_holiday(self, market_code: str, d: date) -> bool:
        """移除节假日"""
        calendar = self.calendars.get(market_code)
        if not calendar:
            return False

        if d in calendar.holidays:
            del calendar.holidays[d]
            return True
        return False

    def on_session_change(self, callback: Callable[[str, SessionType, SessionType], None]):
        """注册会话变更回调"""
        self._session_callbacks.append(callback)

    def on_status_change(self, callback: Callable[[str, MarketStatus], None]):
        """注册状态变更回调"""
        self._status_callbacks.append(callback)

    async def start_monitoring(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Session monitoring started")

    async def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Session monitoring stopped")

    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                for market_code in self.calendars:
                    new_status = self.get_market_status(market_code)

                    with self._lock:
                        old_status = self._status_cache.get(market_code)

                        if old_status:
                            # 检查状态变化
                            if old_status.market_status != new_status.market_status:
                                for callback in self._status_callbacks:
                                    try:
                                        callback(market_code, new_status.market_status)
                                    except Exception as e:
                                        logger.error(f"Status callback error: {e}")

                            # 检查会话变化
                            if old_status.current_session != new_status.current_session:
                                for callback in self._session_callbacks:
                                    try:
                                        callback(
                                            market_code,
                                            old_status.current_session,
                                            new_status.current_session,
                                        )
                                    except Exception as e:
                                        logger.error(f"Session callback error: {e}")

                        self._status_cache[market_code] = new_status

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    def get_all_status(self) -> Dict[str, SessionStatus]:
        """获取所有市场状态"""
        return {code: self.get_market_status(code) for code in self.calendars}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        all_status = self.get_all_status()

        open_markets = sum(1 for s in all_status.values() if s.market_status == MarketStatus.OPEN)
        closed_markets = sum(1 for s in all_status.values() if s.market_status == MarketStatus.CLOSED)

        return {
            "total_markets": len(self.calendars),
            "open_markets": open_markets,
            "closed_markets": closed_markets,
            "monitoring": self._running,
            "market_status": {code: s.to_dict() for code, s in all_status.items()},
        }


# ==================== 工厂函数 ====================

def create_session_manager(
    include_us: bool = True,
    include_hk: bool = True,
    include_cn: bool = True,
    include_futures: bool = True,
    include_forex: bool = True,
    include_crypto: bool = True,
) -> TradingSessionManager:
    """创建会话管理器"""
    manager = TradingSessionManager()

    # 移除不需要的市场
    if not include_us:
        manager.unregister_market("US")
    if not include_hk:
        manager.unregister_market("HK")
    if not include_cn:
        manager.unregister_market("CN")
    if not include_futures:
        manager.unregister_market("CME")
    if not include_forex:
        manager.unregister_market("FX")
    if not include_crypto:
        manager.unregister_market("CRYPTO")

    return manager


def create_market_calendar(
    market_code: str,
    market_name: str,
    market_type: str,
    timezone: str,
    sessions: List[Dict[str, Any]],
    trading_days: List[int] = None,
    currency: str = "USD",
) -> MarketCalendar:
    """创建市场日历"""
    session_list = []
    for s in sessions:
        session_list.append(TradingSession(
            session_type=SessionType(s["session_type"]),
            start_time=time.fromisoformat(s["start_time"]),
            end_time=time.fromisoformat(s["end_time"]),
            timezone=timezone,
            allow_orders=s.get("allow_orders", True),
            allow_market_orders=s.get("allow_market_orders", True),
            allow_limit_orders=s.get("allow_limit_orders", True),
            description=s.get("description", ""),
        ))

    return MarketCalendar(
        market_code=market_code,
        market_name=market_name,
        market_type=MarketType(market_type),
        timezone=timezone,
        sessions=session_list,
        trading_days=set(trading_days) if trading_days else {0, 1, 2, 3, 4},
        currency=currency,
    )


def create_holiday(
    d: str,
    name: str,
    holiday_type: str = "full_day",
    early_close_time: Optional[str] = None,
) -> Holiday:
    """创建节假日"""
    return Holiday(
        date=date.fromisoformat(d),
        name=name,
        holiday_type=HolidayType(holiday_type),
        early_close_time=time.fromisoformat(early_close_time) if early_close_time else None,
    )


def quick_market_check(market_code: str) -> Dict[str, Any]:
    """快速市场检查"""
    manager = TradingSessionManager()
    status = manager.get_market_status(market_code)
    validation = manager.validate_order(market_code)

    return {
        "market_code": market_code,
        "status": status.to_dict(),
        "can_trade": validation.is_valid,
        "validation": validation.to_dict(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'MarketType',
    'SessionType',
    'MarketStatus',
    'HolidayType',
    'OrderSessionRule',
    # 数据类
    'TradingSession',
    'Holiday',
    'MarketCalendar',
    'SessionStatus',
    'OrderValidation',
    'ManagerConfig',
    # 预定义日历
    'create_us_stock_calendar',
    'create_hk_stock_calendar',
    'create_cn_stock_calendar',
    'create_futures_calendar',
    'create_forex_calendar',
    'create_crypto_calendar',
    # 管理器
    'TradingSessionManager',
    # 工厂函数
    'create_session_manager',
    'create_market_calendar',
    'create_holiday',
    'quick_market_check',
]
