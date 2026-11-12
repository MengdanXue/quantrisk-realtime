"""
交易会话管理器测试
"""

import pytest
import asyncio
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock

from quantrisk_realtime.realtime.trading_session_manager import (
    # 枚举
    MarketType,
    SessionType,
    MarketStatus,
    HolidayType,
    OrderSessionRule,
    # 数据类
    TradingSession,
    Holiday,
    MarketCalendar,
    SessionStatus,
    OrderValidation,
    ManagerConfig,
    # 预定义日历
    create_us_stock_calendar,
    create_hk_stock_calendar,
    create_cn_stock_calendar,
    create_futures_calendar,
    create_forex_calendar,
    create_crypto_calendar,
    # 管理器
    TradingSessionManager,
    # 工厂函数
    create_session_manager,
    create_market_calendar,
    create_holiday,
    quick_market_check,
)


# ==================== 枚举测试 ====================

class TestMarketType:
    """市场类型测试"""

    def test_market_type_values(self):
        """测试市场类型值"""
        assert MarketType.STOCK.value == "stock"
        assert MarketType.FUTURES.value == "futures"
        assert MarketType.FOREX.value == "forex"
        assert MarketType.CRYPTO.value == "crypto"


class TestSessionType:
    """交易时段类型测试"""

    def test_session_type_values(self):
        """测试时段类型值"""
        assert SessionType.PRE_MARKET.value == "pre_market"
        assert SessionType.REGULAR.value == "regular"
        assert SessionType.POST_MARKET.value == "post_market"
        assert SessionType.LUNCH_BREAK.value == "lunch_break"
        assert SessionType.CLOSED.value == "closed"


class TestMarketStatus:
    """市场状态测试"""

    def test_market_status_values(self):
        """测试市场状态值"""
        assert MarketStatus.OPEN.value == "open"
        assert MarketStatus.CLOSED.value == "closed"
        assert MarketStatus.HOLIDAY.value == "holiday"
        assert MarketStatus.WEEKEND.value == "weekend"


class TestHolidayType:
    """节假日类型测试"""

    def test_holiday_type_values(self):
        """测试节假日类型值"""
        assert HolidayType.FULL_DAY.value == "full_day"
        assert HolidayType.EARLY_CLOSE.value == "early_close"


# ==================== 数据类测试 ====================

class TestTradingSession:
    """交易时段测试"""

    def test_session_creation(self):
        """测试时段创建"""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(16, 0),
            timezone="America/New_York",
        )

        assert session.session_type == SessionType.REGULAR
        assert session.start_time == time(9, 30)
        assert session.end_time == time(16, 0)

    def test_session_contains_time(self):
        """测试时间包含检查"""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(16, 0),
        )

        assert session.contains_time(time(10, 0)) is True
        assert session.contains_time(time(9, 30)) is True
        assert session.contains_time(time(16, 0)) is True
        assert session.contains_time(time(8, 0)) is False
        assert session.contains_time(time(17, 0)) is False

    def test_session_overnight(self):
        """测试跨午夜时段"""
        session = TradingSession(
            session_type=SessionType.OVERNIGHT,
            start_time=time(18, 0),
            end_time=time(8, 0),
        )

        assert session.contains_time(time(20, 0)) is True
        assert session.contains_time(time(2, 0)) is True
        assert session.contains_time(time(12, 0)) is False

    def test_session_to_dict(self):
        """测试转换为字典"""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(16, 0),
        )

        d = session.to_dict()

        assert d["session_type"] == "regular"
        assert d["start_time"] == "09:30:00"
        assert d["end_time"] == "16:00:00"


class TestHoliday:
    """节假日测试"""

    def test_holiday_creation(self):
        """测试节假日创建"""
        holiday = Holiday(
            date=date(2024, 12, 25),
            name="Christmas Day",
            holiday_type=HolidayType.FULL_DAY,
        )

        assert holiday.date == date(2024, 12, 25)
        assert holiday.name == "Christmas Day"
        assert holiday.holiday_type == HolidayType.FULL_DAY

    def test_holiday_early_close(self):
        """测试提前收盘"""
        holiday = Holiday(
            date=date(2024, 11, 29),
            name="Day After Thanksgiving",
            holiday_type=HolidayType.EARLY_CLOSE,
            early_close_time=time(13, 0),
        )

        assert holiday.holiday_type == HolidayType.EARLY_CLOSE
        assert holiday.early_close_time == time(13, 0)

    def test_holiday_to_dict(self):
        """测试转换为字典"""
        holiday = Holiday(
            date=date(2024, 12, 25),
            name="Christmas Day",
        )

        d = holiday.to_dict()

        assert d["date"] == "2024-12-25"
        assert d["name"] == "Christmas Day"


class TestMarketCalendar:
    """市场日历测试"""

    @pytest.fixture
    def us_calendar(self):
        return create_us_stock_calendar()

    def test_calendar_creation(self, us_calendar):
        """测试日历创建"""
        assert us_calendar.market_code == "US"
        assert us_calendar.market_type == MarketType.STOCK
        assert len(us_calendar.sessions) == 3

    def test_is_trading_day_weekday(self, us_calendar):
        """测试工作日交易日检查"""
        # 2024年12月23日是周一
        monday = date(2024, 12, 23)
        assert us_calendar.is_trading_day(monday) is True

    def test_is_trading_day_weekend(self, us_calendar):
        """测试周末非交易日"""
        # 2024年12月21日是周六
        saturday = date(2024, 12, 21)
        assert us_calendar.is_trading_day(saturday) is False

    def test_is_trading_day_holiday(self, us_calendar):
        """测试节假日非交易日"""
        christmas = date(2024, 12, 25)
        assert us_calendar.is_trading_day(christmas) is False

    def test_get_holiday(self, us_calendar):
        """测试获取节假日"""
        holiday = us_calendar.get_holiday(date(2024, 12, 25))
        assert holiday is not None
        assert holiday.name == "Christmas Day"

    def test_get_current_session(self, us_calendar):
        """测试获取当前时段"""
        # 创建一个常规交易时段内的时间
        dt = datetime(2024, 12, 23, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        session = us_calendar.get_current_session(dt)

        assert session is not None
        assert session.session_type == SessionType.REGULAR

    def test_calendar_to_dict(self, us_calendar):
        """测试日历转换为字典"""
        d = us_calendar.to_dict()

        assert d["market_code"] == "US"
        assert d["market_type"] == "stock"
        assert len(d["sessions"]) == 3


class TestSessionStatus:
    """会话状态测试"""

    def test_status_creation(self):
        """测试状态创建"""
        status = SessionStatus(
            market_code="US",
            market_status=MarketStatus.OPEN,
            current_session=SessionType.REGULAR,
            next_session=SessionType.POST_MARKET,
            next_session_time=datetime.now(),
            is_trading_day=True,
            holiday=None,
        )

        assert status.market_code == "US"
        assert status.market_status == MarketStatus.OPEN
        assert status.current_session == SessionType.REGULAR

    def test_status_to_dict(self):
        """测试状态转换为字典"""
        status = SessionStatus(
            market_code="US",
            market_status=MarketStatus.OPEN,
            current_session=SessionType.REGULAR,
            next_session=None,
            next_session_time=None,
            is_trading_day=True,
            holiday=None,
        )

        d = status.to_dict()

        assert d["market_code"] == "US"
        assert d["market_status"] == "open"
        assert d["current_session"] == "regular"


class TestOrderValidation:
    """订单验证测试"""

    def test_validation_valid(self):
        """测试有效订单"""
        validation = OrderValidation(
            is_valid=True,
            market_code="US",
            session_type=SessionType.REGULAR,
            reason="Order validated",
            allows_market_orders=True,
            allows_limit_orders=True,
        )

        assert validation.is_valid is True
        assert validation.allows_market_orders is True

    def test_validation_invalid(self):
        """测试无效订单"""
        validation = OrderValidation(
            is_valid=False,
            market_code="US",
            session_type=None,
            reason="Market is closed",
        )

        assert validation.is_valid is False

    def test_validation_to_dict(self):
        """测试转换为字典"""
        validation = OrderValidation(
            is_valid=True,
            market_code="US",
            session_type=SessionType.REGULAR,
        )

        d = validation.to_dict()

        assert d["is_valid"] is True
        assert d["session_type"] == "regular"


# ==================== 预定义日历测试 ====================

class TestPredefinedCalendars:
    """预定义日历测试"""

    def test_us_stock_calendar(self):
        """测试美股日历"""
        calendar = create_us_stock_calendar()

        assert calendar.market_code == "US"
        assert calendar.timezone == "America/New_York"
        assert len(calendar.sessions) == 3  # pre, regular, post

    def test_hk_stock_calendar(self):
        """测试港股日历"""
        calendar = create_hk_stock_calendar()

        assert calendar.market_code == "HK"
        assert calendar.timezone == "Asia/Hong_Kong"
        assert len(calendar.sessions) == 5  # open auction, am, lunch, pm, close auction

    def test_cn_stock_calendar(self):
        """测试A股日历"""
        calendar = create_cn_stock_calendar()

        assert calendar.market_code == "CN"
        assert calendar.timezone == "Asia/Shanghai"

    def test_futures_calendar(self):
        """测试期货日历"""
        calendar = create_futures_calendar()

        assert calendar.market_code == "CME"
        assert calendar.market_type == MarketType.FUTURES

    def test_forex_calendar(self):
        """测试外汇日历"""
        calendar = create_forex_calendar()

        assert calendar.market_code == "FX"
        assert calendar.market_type == MarketType.FOREX

    def test_crypto_calendar(self):
        """测试加密货币日历"""
        calendar = create_crypto_calendar()

        assert calendar.market_code == "CRYPTO"
        assert calendar.market_type == MarketType.CRYPTO
        # 加密货币每天都交易
        assert len(calendar.trading_days) == 7


# ==================== 会话管理器测试 ====================

class TestTradingSessionManager:
    """交易会话管理器测试"""

    @pytest.fixture
    def manager(self):
        return TradingSessionManager()

    def test_manager_creation(self, manager):
        """测试管理器创建"""
        assert len(manager.calendars) == 6  # US, HK, CN, CME, FX, CRYPTO

    def test_get_all_markets(self, manager):
        """测试获取所有市场"""
        markets = manager.get_all_markets()

        assert "US" in markets
        assert "HK" in markets
        assert "CRYPTO" in markets

    def test_register_market(self, manager):
        """测试注册市场"""
        calendar = MarketCalendar(
            market_code="TEST",
            market_name="Test Market",
            market_type=MarketType.STOCK,
            timezone="UTC",
            sessions=[],
        )

        manager.register_market(calendar)

        assert "TEST" in manager.calendars

    def test_unregister_market(self, manager):
        """测试注销市场"""
        result = manager.unregister_market("CRYPTO")

        assert result is True
        assert "CRYPTO" not in manager.calendars

    def test_unregister_nonexistent(self, manager):
        """测试注销不存在的市场"""
        result = manager.unregister_market("NONEXISTENT")
        assert result is False

    def test_get_calendar(self, manager):
        """测试获取日历"""
        calendar = manager.get_calendar("US")

        assert calendar is not None
        assert calendar.market_code == "US"

    def test_get_calendar_nonexistent(self, manager):
        """测试获取不存在的日历"""
        calendar = manager.get_calendar("NONEXISTENT")
        assert calendar is None

    def test_get_market_status(self, manager):
        """测试获取市场状态"""
        status = manager.get_market_status("US")

        assert status.market_code == "US"
        assert status.market_status in MarketStatus

    def test_get_market_status_nonexistent(self, manager):
        """测试获取不存在市场的状态"""
        status = manager.get_market_status("NONEXISTENT")

        assert status.market_status == MarketStatus.CLOSED
        assert status.is_trading_day is False

    def test_get_market_status_specific_time(self, manager):
        """测试指定时间的市场状态"""
        # 2024年12月23日10:30 (美东时间，周一)
        dt = datetime(2024, 12, 23, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        status = manager.get_market_status("US", dt)

        assert status.current_session == SessionType.REGULAR
        assert status.is_trading_day is True

    def test_get_market_status_premarket(self, manager):
        """测试盘前时段状态"""
        # 2024年12月23日7:00 (美东时间)
        dt = datetime(2024, 12, 23, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        status = manager.get_market_status("US", dt)

        assert status.current_session == SessionType.PRE_MARKET
        assert status.market_status == MarketStatus.PRE_OPEN

    def test_get_market_status_weekend(self, manager):
        """测试周末状态"""
        # 2024年12月21日是周六
        dt = datetime(2024, 12, 21, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        status = manager.get_market_status("US", dt)

        assert status.market_status == MarketStatus.WEEKEND
        assert status.is_trading_day is False

    def test_get_market_status_holiday(self, manager):
        """测试节假日状态"""
        # 圣诞节
        dt = datetime(2024, 12, 25, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        status = manager.get_market_status("US", dt)

        assert status.market_status == MarketStatus.HOLIDAY
        assert status.holiday == "Christmas Day"

    def test_validate_order_valid(self, manager):
        """测试有效订单验证"""
        # 常规交易时段
        dt = datetime(2024, 12, 23, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        validation = manager.validate_order("US", "limit", OrderSessionRule.REGULAR_ONLY, dt)

        assert validation.is_valid is True
        assert validation.allows_limit_orders is True

    def test_validate_order_wrong_session(self, manager):
        """测试错误时段订单验证"""
        # 盘前时段，但要求常规时段
        dt = datetime(2024, 12, 23, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        validation = manager.validate_order("US", "limit", OrderSessionRule.REGULAR_ONLY, dt)

        assert validation.is_valid is False

    def test_validate_order_market_order_premarket(self, manager):
        """测试盘前市价单"""
        # 盘前时段不允许市价单
        dt = datetime(2024, 12, 23, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        validation = manager.validate_order("US", "market", OrderSessionRule.EXTENDED_HOURS, dt)

        assert validation.is_valid is False
        assert "Market orders not allowed" in validation.reason

    def test_validate_order_holiday(self, manager):
        """测试节假日订单验证"""
        dt = datetime(2024, 12, 25, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        validation = manager.validate_order("US", "limit", dt=dt)

        assert validation.is_valid is False
        assert "trading day" in validation.reason.lower()

    def test_is_market_open(self, manager):
        """测试市场开放检查"""
        dt = datetime(2024, 12, 23, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert manager.is_market_open("US", dt) is True

        dt = datetime(2024, 12, 21, 10, 0, tzinfo=ZoneInfo("America/New_York"))  # 周六
        assert manager.is_market_open("US", dt) is False

    def test_is_regular_hours(self, manager):
        """测试常规时段检查"""
        dt = datetime(2024, 12, 23, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert manager.is_regular_hours("US", dt) is True

        dt = datetime(2024, 12, 23, 7, 0, tzinfo=ZoneInfo("America/New_York"))  # 盘前
        assert manager.is_regular_hours("US", dt) is False

    def test_get_trading_days(self, manager):
        """测试获取交易日"""
        start = date(2024, 12, 23)  # 周一
        end = date(2024, 12, 27)    # 周五

        trading_days = manager.get_trading_days("US", start, end)

        # 12/23, 12/24, 12/26, 12/27 (12/25是圣诞节)
        assert len(trading_days) == 4
        assert date(2024, 12, 25) not in trading_days

    def test_get_holidays(self, manager):
        """测试获取节假日"""
        holidays = manager.get_holidays("US", 2024)

        assert len(holidays) > 0
        holiday_names = [h.name for h in holidays]
        assert "Christmas Day" in holiday_names

    def test_add_holiday(self, manager):
        """测试添加节假日"""
        holiday = Holiday(
            date=date(2024, 12, 31),
            name="Test Holiday",
        )

        result = manager.add_holiday("US", holiday)

        assert result is True
        assert date(2024, 12, 31) in manager.calendars["US"].holidays

    def test_remove_holiday(self, manager):
        """测试移除节假日"""
        result = manager.remove_holiday("US", date(2024, 12, 25))

        assert result is True
        assert date(2024, 12, 25) not in manager.calendars["US"].holidays

    def test_on_session_change(self, manager):
        """测试会话变更回调"""
        callback = Mock()
        manager.on_session_change(callback)

        assert callback in manager._session_callbacks

    def test_on_status_change(self, manager):
        """测试状态变更回调"""
        callback = Mock()
        manager.on_status_change(callback)

        assert callback in manager._status_callbacks

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, manager):
        """测试启动和停止监控"""
        await manager.start_monitoring()
        assert manager._running is True

        await manager.stop_monitoring()
        assert manager._running is False

    def test_get_all_status(self, manager):
        """测试获取所有状态"""
        all_status = manager.get_all_status()

        assert len(all_status) == 6
        assert "US" in all_status
        assert "CRYPTO" in all_status

    def test_get_statistics(self, manager):
        """测试获取统计"""
        stats = manager.get_statistics()

        assert stats["total_markets"] == 6
        assert "market_status" in stats


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_session_manager(self):
        """测试创建会话管理器"""
        manager = create_session_manager(
            include_us=True,
            include_hk=False,
            include_cn=False,
            include_futures=False,
            include_forex=False,
            include_crypto=True,
        )

        assert "US" in manager.calendars
        assert "CRYPTO" in manager.calendars
        assert "HK" not in manager.calendars

    def test_create_market_calendar(self):
        """测试创建市场日历"""
        sessions = [
            {
                "session_type": "regular",
                "start_time": "09:00:00",
                "end_time": "16:00:00",
            }
        ]

        calendar = create_market_calendar(
            market_code="TEST",
            market_name="Test Market",
            market_type="stock",
            timezone="UTC",
            sessions=sessions,
        )

        assert calendar.market_code == "TEST"
        assert len(calendar.sessions) == 1

    def test_create_holiday(self):
        """测试创建节假日"""
        holiday = create_holiday(
            d="2024-12-25",
            name="Christmas",
            holiday_type="full_day",
        )

        assert holiday.date == date(2024, 12, 25)
        assert holiday.name == "Christmas"

    def test_create_holiday_early_close(self):
        """测试创建提前收盘节假日"""
        holiday = create_holiday(
            d="2024-11-29",
            name="Black Friday",
            holiday_type="early_close",
            early_close_time="13:00:00",
        )

        assert holiday.holiday_type == HolidayType.EARLY_CLOSE
        assert holiday.early_close_time == time(13, 0)

    def test_quick_market_check(self):
        """测试快速市场检查"""
        result = quick_market_check("US")

        assert result["market_code"] == "US"
        assert "status" in result
        assert "can_trade" in result
        assert "validation" in result


# ==================== 边缘情况测试 ====================

class TestEdgeCases:
    """边缘情况测试"""

    def test_crypto_always_open(self):
        """测试加密货币始终开放"""
        manager = TradingSessionManager()

        # 周末
        dt = datetime(2024, 12, 21, 10, 0, tzinfo=ZoneInfo("UTC"))
        status = manager.get_market_status("CRYPTO", dt)

        assert status.market_status == MarketStatus.OPEN

    def test_validate_unknown_market(self):
        """测试验证未知市场"""
        manager = TradingSessionManager()
        validation = manager.validate_order("UNKNOWN")

        assert validation.is_valid is False
        assert "Unknown market" in validation.reason

    def test_empty_sessions(self):
        """测试空时段日历"""
        calendar = MarketCalendar(
            market_code="EMPTY",
            market_name="Empty Market",
            market_type=MarketType.STOCK,
            timezone="UTC",
            sessions=[],
        )

        manager = TradingSessionManager()
        manager.register_market(calendar)

        status = manager.get_market_status("EMPTY")
        assert status.current_session is None

    def test_midnight_crossing_session(self):
        """测试跨午夜时段"""
        session = TradingSession(
            session_type=SessionType.OVERNIGHT,
            start_time=time(22, 0),
            end_time=time(6, 0),
        )

        assert session.contains_time(time(23, 0)) is True
        assert session.contains_time(time(3, 0)) is True
        assert session.contains_time(time(12, 0)) is False


# ==================== 序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    def test_session_serialization(self):
        """测试时段序列化"""
        session = TradingSession(
            session_type=SessionType.REGULAR,
            start_time=time(9, 30),
            end_time=time(16, 0),
            description="Regular hours",
        )

        d = session.to_dict()

        assert d["session_type"] == "regular"
        assert d["description"] == "Regular hours"

    def test_holiday_serialization(self):
        """测试节假日序列化"""
        holiday = Holiday(
            date=date(2024, 12, 25),
            name="Christmas",
            holiday_type=HolidayType.FULL_DAY,
        )

        d = holiday.to_dict()

        assert d["date"] == "2024-12-25"
        assert d["holiday_type"] == "full_day"

    def test_calendar_serialization(self):
        """测试日历序列化"""
        calendar = create_us_stock_calendar()
        d = calendar.to_dict()

        assert d["market_code"] == "US"
        assert len(d["sessions"]) == 3

    def test_status_serialization(self):
        """测试状态序列化"""
        status = SessionStatus(
            market_code="US",
            market_status=MarketStatus.OPEN,
            current_session=SessionType.REGULAR,
            next_session=SessionType.POST_MARKET,
            next_session_time=datetime.now(),
            is_trading_day=True,
            holiday=None,
        )

        d = status.to_dict()

        assert d["market_status"] == "open"
        assert d["current_session"] == "regular"
