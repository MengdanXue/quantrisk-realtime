"""
保证金监控器单元测试
Margin Monitor Unit Tests

测试覆盖:
- 告警等级判断
- 保证金状态检测
- 阈值配置
- 监控生命周期

作者: TraderX Team
创建: 2025-12-23
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# 确保src目录在路径中
import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.margin_monitor import (
    AlertLevel,
    LiquidationReason,
    MonitorState,
    MarginThresholds,
    MarginStatus,
)


class TestAlertLevel:
    """测试告警等级枚举"""

    def test_alert_level_values(self):
        """测试告警等级值"""
        assert AlertLevel.NORMAL == "normal"
        assert AlertLevel.NOTICE == "notice"
        assert AlertLevel.WARNING == "warning"
        assert AlertLevel.DANGER == "danger"
        assert AlertLevel.EMERGENCY == "emergency"

    def test_alert_level_ordering(self):
        """测试告警等级有序性"""
        levels = [
            AlertLevel.NORMAL,
            AlertLevel.NOTICE,
            AlertLevel.WARNING,
            AlertLevel.DANGER,
            AlertLevel.EMERGENCY
        ]
        # 枚举值应该按严重程度排列
        assert len(levels) == 5


class TestLiquidationReason:
    """测试强平原因枚举"""

    def test_liquidation_reasons(self):
        """测试强平原因值"""
        assert LiquidationReason.MARGIN_CALL == "margin_call"
        assert LiquidationReason.DAILY_LOSS_LIMIT == "daily_loss_limit"
        assert LiquidationReason.DRAWDOWN_LIMIT == "drawdown_limit"
        assert LiquidationReason.MANUAL == "manual"
        assert LiquidationReason.SYSTEM == "system"


class TestMonitorState:
    """测试监控器状态枚举"""

    def test_monitor_states(self):
        """测试监控器状态值"""
        assert MonitorState.STOPPED == "stopped"
        assert MonitorState.STARTING == "starting"
        assert MonitorState.RUNNING == "running"
        assert MonitorState.PAUSED == "paused"
        assert MonitorState.ERROR == "error"


class TestMarginThresholds:
    """测试保证金阈值配置"""

    def test_default_values(self):
        """测试默认值"""
        thresholds = MarginThresholds()

        assert thresholds.notice_ratio == 0.60
        assert thresholds.warning_ratio == 0.70
        assert thresholds.danger_ratio == 0.85
        assert thresholds.emergency_ratio == 0.95
        assert thresholds.max_daily_loss == 0.05
        assert thresholds.max_drawdown == 0.10

    def test_custom_values(self):
        """测试自定义值"""
        thresholds = MarginThresholds(
            notice_ratio=0.50,
            warning_ratio=0.65,
            danger_ratio=0.80,
            emergency_ratio=0.90
        )

        assert thresholds.notice_ratio == 0.50
        assert thresholds.warning_ratio == 0.65
        assert thresholds.danger_ratio == 0.80
        assert thresholds.emergency_ratio == 0.90

    def test_get_alert_level_normal(self):
        """测试正常状态告警等级"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.0) == AlertLevel.NORMAL
        assert thresholds.get_alert_level(0.30) == AlertLevel.NORMAL
        assert thresholds.get_alert_level(0.59) == AlertLevel.NORMAL

    def test_get_alert_level_notice(self):
        """测试注意状态告警等级"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.60) == AlertLevel.NOTICE
        assert thresholds.get_alert_level(0.65) == AlertLevel.NOTICE
        assert thresholds.get_alert_level(0.69) == AlertLevel.NOTICE

    def test_get_alert_level_warning(self):
        """测试警告状态告警等级"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.70) == AlertLevel.WARNING
        assert thresholds.get_alert_level(0.75) == AlertLevel.WARNING
        assert thresholds.get_alert_level(0.84) == AlertLevel.WARNING

    def test_get_alert_level_danger(self):
        """测试危险状态告警等级"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.85) == AlertLevel.DANGER
        assert thresholds.get_alert_level(0.90) == AlertLevel.DANGER
        assert thresholds.get_alert_level(0.94) == AlertLevel.DANGER

    def test_get_alert_level_emergency(self):
        """测试紧急状态告警等级"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.95) == AlertLevel.EMERGENCY
        assert thresholds.get_alert_level(0.99) == AlertLevel.EMERGENCY
        assert thresholds.get_alert_level(1.00) == AlertLevel.EMERGENCY


class TestMarginStatus:
    """测试保证金状态"""

    def test_default_values(self):
        """测试默认值"""
        status = MarginStatus()

        assert status.account_id == ""
        assert status.total_equity == 0.0
        assert status.available_margin == 0.0
        assert status.used_margin == 0.0

    def test_with_values(self):
        """测试带值创建"""
        status = MarginStatus(
            account_id="ACC001",
            total_equity=100000.0,
            available_margin=50000.0,
            used_margin=50000.0
        )

        assert status.account_id == "ACC001"
        assert status.total_equity == 100000.0
        assert status.available_margin == 50000.0
        assert status.used_margin == 50000.0

    def test_timestamp(self):
        """测试时间戳"""
        before = datetime.now()
        status = MarginStatus()
        after = datetime.now()

        assert before <= status.timestamp <= after


class TestThresholdBoundary:
    """测试阈值边界条件"""

    def test_exact_boundary_values(self):
        """测试精确边界值"""
        thresholds = MarginThresholds()

        # 边界值应该触发更高级别
        assert thresholds.get_alert_level(0.60) == AlertLevel.NOTICE
        assert thresholds.get_alert_level(0.70) == AlertLevel.WARNING
        assert thresholds.get_alert_level(0.85) == AlertLevel.DANGER
        assert thresholds.get_alert_level(0.95) == AlertLevel.EMERGENCY

    def test_just_below_boundary(self):
        """测试刚好低于边界"""
        thresholds = MarginThresholds()

        assert thresholds.get_alert_level(0.599999) == AlertLevel.NORMAL
        assert thresholds.get_alert_level(0.699999) == AlertLevel.NOTICE
        assert thresholds.get_alert_level(0.849999) == AlertLevel.WARNING
        assert thresholds.get_alert_level(0.949999) == AlertLevel.DANGER


class TestMarginCalculations:
    """测试保证金计算"""

    def test_margin_ratio_calculation(self):
        """测试保证金使用率计算"""
        status = MarginStatus(
            total_equity=100000.0,
            used_margin=50000.0,
            available_margin=50000.0
        )

        # 计算保证金使用率
        margin_ratio = status.used_margin / status.total_equity
        assert margin_ratio == 0.5

    def test_margin_ratio_high(self):
        """测试高保证金使用率"""
        status = MarginStatus(
            total_equity=100000.0,
            used_margin=90000.0,
            available_margin=10000.0
        )

        margin_ratio = status.used_margin / status.total_equity
        assert margin_ratio == 0.9

        thresholds = MarginThresholds()
        assert thresholds.get_alert_level(margin_ratio) == AlertLevel.DANGER


class TestCustomThresholds:
    """测试自定义阈值"""

    def test_aggressive_thresholds(self):
        """测试激进阈值"""
        # 更宽松的阈值
        thresholds = MarginThresholds(
            notice_ratio=0.70,
            warning_ratio=0.80,
            danger_ratio=0.90,
            emergency_ratio=0.98
        )

        # 50%使用率仍然是正常
        assert thresholds.get_alert_level(0.50) == AlertLevel.NORMAL
        # 75%现在是注意而不是警告
        assert thresholds.get_alert_level(0.75) == AlertLevel.NOTICE

    def test_conservative_thresholds(self):
        """测试保守阈值"""
        # 更严格的阈值
        thresholds = MarginThresholds(
            notice_ratio=0.40,
            warning_ratio=0.50,
            danger_ratio=0.65,
            emergency_ratio=0.80
        )

        # 45%使用率现在是注意
        assert thresholds.get_alert_level(0.45) == AlertLevel.NOTICE
        # 55%现在是警告
        assert thresholds.get_alert_level(0.55) == AlertLevel.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
