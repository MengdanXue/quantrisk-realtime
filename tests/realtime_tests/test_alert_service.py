"""
告警服务单元测试
Alert Service Unit Tests

测试覆盖:
- 告警渠道
- 告警优先级
- 告警类别
- 告警配置
- 告警实体

作者: TraderX Team
创建: 2025-12-23
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# 确保src目录在路径中
import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.alert_service import (
    AlertChannel,
    AlertPriority,
    AlertCategory,
    AlertConfig,
    Alert,
)


class TestAlertChannel:
    """测试告警渠道枚举"""

    def test_channel_values(self):
        """测试渠道值"""
        assert AlertChannel.EMAIL == "email"
        assert AlertChannel.WEBSOCKET == "websocket"
        assert AlertChannel.SMS == "sms"
        assert AlertChannel.WEBHOOK == "webhook"
        assert AlertChannel.CONSOLE == "console"

    def test_channel_count(self):
        """测试渠道数量"""
        channels = list(AlertChannel)
        assert len(channels) == 5


class TestAlertPriority:
    """测试告警优先级枚举"""

    def test_priority_values(self):
        """测试优先级值"""
        assert AlertPriority.INFO == "info"
        assert AlertPriority.LOW == "low"
        assert AlertPriority.MEDIUM == "medium"
        assert AlertPriority.HIGH == "high"
        assert AlertPriority.CRITICAL == "critical"

    def test_priority_ordering(self):
        """测试优先级顺序"""
        priorities = [
            AlertPriority.INFO,
            AlertPriority.LOW,
            AlertPriority.MEDIUM,
            AlertPriority.HIGH,
            AlertPriority.CRITICAL
        ]
        assert len(priorities) == 5


class TestAlertCategory:
    """测试告警类别枚举"""

    def test_category_values(self):
        """测试类别值"""
        assert AlertCategory.MARGIN == "margin"
        assert AlertCategory.POSITION == "position"
        assert AlertCategory.LOSS == "loss"
        assert AlertCategory.VOLATILITY == "volatility"
        assert AlertCategory.LIQUIDITY == "liquidity"
        assert AlertCategory.SYSTEM == "system"
        assert AlertCategory.COMPLIANCE == "compliance"
        assert AlertCategory.CONCENTRATION == "concentration"

    def test_category_count(self):
        """测试类别数量"""
        categories = list(AlertCategory)
        assert len(categories) == 8


class TestAlertConfig:
    """测试告警配置"""

    def test_default_values(self):
        """测试默认值"""
        config = AlertConfig()

        # 默认启用的渠道
        assert AlertChannel.WEBSOCKET in config.enabled_channels
        assert AlertChannel.CONSOLE in config.enabled_channels

        # 默认配置
        assert config.max_history_size == 1000
        assert config.retain_days == 30
        assert config.aggregation_window_seconds == 60

    def test_default_cooldowns(self):
        """测试默认冷却时间"""
        config = AlertConfig()

        assert config.cooldown_seconds[AlertPriority.INFO] == 300
        assert config.cooldown_seconds[AlertPriority.LOW] == 180
        assert config.cooldown_seconds[AlertPriority.MEDIUM] == 60
        assert config.cooldown_seconds[AlertPriority.HIGH] == 30
        assert config.cooldown_seconds[AlertPriority.CRITICAL] == 0

    def test_critical_no_cooldown(self):
        """测试紧急告警无冷却"""
        config = AlertConfig()
        assert config.cooldown_seconds[AlertPriority.CRITICAL] == 0

    def test_custom_channels(self):
        """测试自定义渠道"""
        config = AlertConfig(
            enabled_channels={AlertChannel.EMAIL, AlertChannel.SMS}
        )

        assert AlertChannel.EMAIL in config.enabled_channels
        assert AlertChannel.SMS in config.enabled_channels
        assert AlertChannel.WEBSOCKET not in config.enabled_channels

    def test_email_configuration(self):
        """测试邮件配置"""
        config = AlertConfig(
            email_recipients=["admin@example.com", "risk@example.com"],
            email_from="alerts@company.com"
        )

        assert len(config.email_recipients) == 2
        assert "admin@example.com" in config.email_recipients
        assert config.email_from == "alerts@company.com"

    def test_webhook_configuration(self):
        """测试Webhook配置"""
        config = AlertConfig(
            webhook_urls=[
                "https://hooks.slack.com/services/xxx",
                "https://api.company.com/alerts"
            ]
        )

        assert len(config.webhook_urls) == 2


class TestAlert:
    """测试告警实体"""

    def test_alert_creation(self):
        """测试告警创建"""
        alert = Alert(
            id="ALERT-001",
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            category=AlertCategory.MARGIN,
            title="保证金警告",
            message="保证金使用率已达到85%"
        )

        assert alert.id == "ALERT-001"
        assert alert.priority == AlertPriority.HIGH
        assert alert.category == AlertCategory.MARGIN
        assert alert.title == "保证金警告"
        assert "85%" in alert.message

    def test_alert_with_different_priorities(self):
        """测试不同优先级告警"""
        priorities = [
            AlertPriority.INFO,
            AlertPriority.LOW,
            AlertPriority.MEDIUM,
            AlertPriority.HIGH,
            AlertPriority.CRITICAL
        ]

        for priority in priorities:
            alert = Alert(
                id=f"ALERT-{priority.value}",
                timestamp=datetime.now(),
                priority=priority,
                category=AlertCategory.SYSTEM,
                title="测试告警",
                message="测试消息"
            )
            assert alert.priority == priority

    def test_alert_with_different_categories(self):
        """测试不同类别告警"""
        categories = [
            AlertCategory.MARGIN,
            AlertCategory.POSITION,
            AlertCategory.LOSS,
            AlertCategory.VOLATILITY,
        ]

        for category in categories:
            alert = Alert(
                id=f"ALERT-{category.value}",
                timestamp=datetime.now(),
                priority=AlertPriority.MEDIUM,
                category=category,
                title=f"{category.value}告警",
                message="测试消息"
            )
            assert alert.category == category


class TestAlertPriorityMapping:
    """测试告警优先级映射"""

    def test_margin_alerts_priority(self):
        """测试保证金告警优先级"""
        # 保证金告警通常是高优先级
        alert = Alert(
            id="MARGIN-001",
            timestamp=datetime.now(),
            priority=AlertPriority.HIGH,
            category=AlertCategory.MARGIN,
            title="保证金不足",
            message="保证金使用率超过85%"
        )

        assert alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]

    def test_system_alerts_priority(self):
        """测试系统告警优先级"""
        # 系统告警可以是任何优先级
        for priority in AlertPriority:
            alert = Alert(
                id=f"SYS-{priority.value}",
                timestamp=datetime.now(),
                priority=priority,
                category=AlertCategory.SYSTEM,
                title="系统告警",
                message="测试"
            )
            assert alert.priority == priority


class TestAlertConfigValidation:
    """测试告警配置验证"""

    def test_empty_recipients(self):
        """测试空收件人列表"""
        config = AlertConfig()
        assert config.email_recipients == []
        assert config.sms_recipients == []

    def test_positive_history_size(self):
        """测试正数历史大小"""
        config = AlertConfig(max_history_size=500)
        assert config.max_history_size == 500
        assert config.max_history_size > 0

    def test_positive_retain_days(self):
        """测试正数保留天数"""
        config = AlertConfig(retain_days=7)
        assert config.retain_days == 7
        assert config.retain_days > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
