"""
合规监控引擎测试 (Compliance Monitor Tests)

测试覆盖:
- 规则管理
- 受限证券管理
- 持仓限制检查
- 集中度检查
- 交易频率检查
- 洗售交易检测
- 违规管理
- 审计日志
- 报告生成
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from quantrisk_realtime.realtime.compliance_monitor import (
    # 枚举
    RuleType,
    RuleSeverity,
    ViolationStatus,
    RestrictionType,
    MonitorState,
    # 数据类
    ComplianceRule,
    TradeForCompliance,
    ComplianceViolation,
    RestrictedSecurity,
    AuditEntry,
    ComplianceCheckResult,
    ComplianceReport,
    ComplianceConfig,
    # 模拟实现
    SimulatedPositionProvider,
    SimulatedMarketDataProvider,
    # 核心引擎
    ComplianceMonitor,
    # 工厂函数
    create_compliance_monitor,
    create_compliance_rule,
    create_trade_for_compliance,
    create_restricted_security,
    quick_compliance_check,
)


# ==================== 测试夹具 ====================

@pytest.fixture
def config():
    """创建测试配置"""
    return ComplianceConfig(
        wash_trade_time_window_seconds=60,
        max_trades_per_minute=10,
        max_position_pct_of_adv=0.1,
        max_concentration_pct=0.2,
    )


@pytest.fixture
def position_provider():
    """创建模拟持仓提供者"""
    provider = SimulatedPositionProvider()
    provider.set_position("ACC001", "AAPL", 1000)
    provider.set_position("ACC001", "GOOGL", 500)
    provider.set_portfolio_value("ACC001", 1000000)
    return provider


@pytest.fixture
def market_data_provider():
    """创建模拟市场数据提供者"""
    provider = SimulatedMarketDataProvider()
    provider.set_price("AAPL", 150.0)
    provider.set_price("GOOGL", 100.0)
    provider.set_adv("AAPL", 10000)
    provider.set_adv("GOOGL", 5000)
    return provider


@pytest.fixture
def monitor(config, position_provider, market_data_provider):
    """创建合规监控引擎"""
    return ComplianceMonitor(
        config=config,
        position_provider=position_provider,
        market_data_provider=market_data_provider,
    )


@pytest.fixture
def sample_trade():
    """创建示例交易"""
    return TradeForCompliance(
        trade_id="T001",
        account_id="ACC001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.0,
        trader_id="TRADER001",
    )


# ==================== 枚举测试 ====================

class TestEnums:
    """枚举测试"""

    def test_rule_type_values(self):
        """测试规则类型枚举值"""
        assert RuleType.POSITION_LIMIT.value == "position_limit"
        assert RuleType.WASH_TRADING.value == "wash_trading"
        assert RuleType.RESTRICTED_SECURITY.value == "restricted_security"

    def test_rule_severity_values(self):
        """测试规则严重性枚举值"""
        assert RuleSeverity.INFO.value == "info"
        assert RuleSeverity.WARNING.value == "warning"
        assert RuleSeverity.VIOLATION.value == "violation"
        assert RuleSeverity.CRITICAL.value == "critical"

    def test_violation_status_values(self):
        """测试违规状态枚举值"""
        assert ViolationStatus.OPEN.value == "open"
        assert ViolationStatus.RESOLVED.value == "resolved"
        assert ViolationStatus.ESCALATED.value == "escalated"

    def test_restriction_type_values(self):
        """测试限制类型枚举值"""
        assert RestrictionType.BLACKLIST.value == "blacklist"
        assert RestrictionType.WHITELIST.value == "whitelist"
        assert RestrictionType.WATCHLIST.value == "watchlist"


# ==================== 数据类测试 ====================

class TestComplianceRule:
    """合规规则测试"""

    def test_create_rule(self):
        """测试创建规则"""
        rule = ComplianceRule(
            rule_id="R001",
            rule_type=RuleType.POSITION_LIMIT,
            name="持仓限制",
            description="限制单一标的持仓",
            condition={"max_pct": 0.1},
        )
        assert rule.rule_id == "R001"
        assert rule.rule_type == RuleType.POSITION_LIMIT
        assert rule.enabled is True

    def test_rule_to_dict(self):
        """测试规则序列化"""
        rule = ComplianceRule(
            rule_id="R001",
            rule_type=RuleType.CONCENTRATION,
            name="集中度限制",
            description="限制集中度",
            condition={"max_pct": 0.2},
            severity=RuleSeverity.WARNING,
        )
        data = rule.to_dict()
        assert data["rule_id"] == "R001"
        assert data["rule_type"] == "concentration"
        assert data["severity"] == "warning"


class TestTradeForCompliance:
    """待检查交易测试"""

    def test_create_trade(self, sample_trade):
        """测试创建交易"""
        assert sample_trade.trade_id == "T001"
        assert sample_trade.symbol == "AAPL"
        assert sample_trade.side == "buy"

    def test_notional_value(self, sample_trade):
        """测试名义价值计算"""
        assert sample_trade.notional_value == 15000.0  # 100 * 150

    def test_trade_to_dict(self, sample_trade):
        """测试交易序列化"""
        data = sample_trade.to_dict()
        assert data["trade_id"] == "T001"
        assert data["notional_value"] == 15000.0


class TestComplianceViolation:
    """合规违规测试"""

    def test_create_violation(self):
        """测试创建违规"""
        violation = ComplianceViolation(
            violation_id="V001",
            rule_id="R001",
            rule_name="持仓限制",
            rule_type=RuleType.POSITION_LIMIT,
            severity=RuleSeverity.VIOLATION,
            description="持仓超限",
        )
        assert violation.violation_id == "V001"
        assert violation.status == ViolationStatus.OPEN

    def test_violation_to_dict(self):
        """测试违规序列化"""
        violation = ComplianceViolation(
            violation_id="V001",
            rule_id="R001",
            rule_name="持仓限制",
            rule_type=RuleType.POSITION_LIMIT,
            severity=RuleSeverity.CRITICAL,
            description="严重违规",
        )
        data = violation.to_dict()
        assert data["severity"] == "critical"
        assert data["status"] == "open"


class TestRestrictedSecurity:
    """受限证券测试"""

    def test_create_restricted_security(self):
        """测试创建受限证券"""
        security = RestrictedSecurity(
            symbol="XYZ",
            restriction_type=RestrictionType.BLACKLIST,
            reason="内幕交易调查",
        )
        assert security.symbol == "XYZ"
        assert security.restriction_type == RestrictionType.BLACKLIST

    def test_is_active(self):
        """测试是否有效"""
        # 永久限制
        security1 = RestrictedSecurity(
            symbol="XYZ",
            restriction_type=RestrictionType.BLACKLIST,
            reason="永久限制",
        )
        assert security1.is_active() is True

        # 过期限制
        security2 = RestrictedSecurity(
            symbol="ABC",
            restriction_type=RestrictionType.WATCHLIST,
            reason="临时限制",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now() - timedelta(days=5),
        )
        assert security2.is_active() is False

        # 未来生效
        security3 = RestrictedSecurity(
            symbol="DEF",
            restriction_type=RestrictionType.GREYLIST,
            reason="未来限制",
            start_date=datetime.now() + timedelta(days=1),
        )
        assert security3.is_active() is False

    def test_restricted_security_to_dict(self):
        """测试受限证券序列化"""
        security = RestrictedSecurity(
            symbol="XYZ",
            restriction_type=RestrictionType.BLACKLIST,
            reason="测试",
            requires_approval=True,
        )
        data = security.to_dict()
        assert data["symbol"] == "XYZ"
        assert data["restriction_type"] == "blacklist"
        assert data["requires_approval"] is True


class TestAuditEntry:
    """审计日志测试"""

    def test_create_audit_entry(self):
        """测试创建审计条目"""
        entry = AuditEntry(
            entry_id="A001",
            action="add_rule",
            actor="admin",
            target_type="rule",
            target_id="R001",
        )
        assert entry.entry_id == "A001"
        assert entry.action == "add_rule"

    def test_audit_entry_to_dict(self):
        """测试审计条目序列化"""
        entry = AuditEntry(
            entry_id="A001",
            action="update_violation",
            actor="compliance_officer",
            target_type="violation",
            target_id="V001",
            old_value={"status": "open"},
            new_value={"status": "resolved"},
        )
        data = entry.to_dict()
        assert data["action"] == "update_violation"
        assert data["old_value"]["status"] == "open"


# ==================== 规则管理测试 ====================

class TestRuleManagement:
    """规则管理测试"""

    def test_add_rule(self, monitor):
        """测试添加规则"""
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="订单大小限制",
            description="限制订单大小",
            condition={"max_order_value": 100000},
        )
        result = monitor.add_rule(rule)
        assert result is True
        assert rule.rule_id in monitor.rules

    def test_add_duplicate_rule(self, monitor):
        """测试添加重复规则"""
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="测试规则",
            description="测试",
            condition={},
            rule_id="duplicate_id",
        )
        monitor.add_rule(rule)
        result = monitor.add_rule(rule)
        assert result is False

    def test_update_rule(self, monitor):
        """测试更新规则"""
        # 获取已有规则
        rules = monitor.get_all_rules()
        rule_id = rules[0].rule_id

        result = monitor.update_rule(rule_id, {"severity": RuleSeverity.CRITICAL})
        assert result is True

        updated_rule = monitor.get_rule(rule_id)
        assert updated_rule.severity == RuleSeverity.CRITICAL

    def test_remove_rule(self, monitor):
        """测试移除规则"""
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="待删除规则",
            description="测试删除",
            condition={},
        )
        monitor.add_rule(rule)
        result = monitor.remove_rule(rule.rule_id)
        assert result is True
        assert rule.rule_id not in monitor.rules

    def test_enable_disable_rule(self, monitor):
        """测试启用/禁用规则"""
        rules = monitor.get_all_rules()
        rule_id = rules[0].rule_id

        monitor.disable_rule(rule_id)
        rule = monitor.get_rule(rule_id)
        assert rule.enabled is False

        monitor.enable_rule(rule_id)
        rule = monitor.get_rule(rule_id)
        assert rule.enabled is True

    def test_get_all_rules_enabled_only(self, monitor):
        """测试获取启用的规则"""
        rules = monitor.get_all_rules()
        rule_id = rules[0].rule_id
        monitor.disable_rule(rule_id)

        all_rules = monitor.get_all_rules(enabled_only=False)
        enabled_rules = monitor.get_all_rules(enabled_only=True)
        assert len(enabled_rules) < len(all_rules)


# ==================== 受限证券管理测试 ====================

class TestRestrictedSecurityManagement:
    """受限证券管理测试"""

    def test_add_restricted_security(self, monitor):
        """测试添加受限证券"""
        security = create_restricted_security(
            symbol="XYZ",
            restriction_type=RestrictionType.BLACKLIST,
            reason="内幕交易",
        )
        result = monitor.add_restricted_security(security)
        assert result is True
        assert "XYZ" in monitor.restricted_securities

    def test_remove_restricted_security(self, monitor):
        """测试移除受限证券"""
        security = create_restricted_security(
            symbol="ABC",
            restriction_type=RestrictionType.WATCHLIST,
            reason="临时观察",
        )
        monitor.add_restricted_security(security)
        result = monitor.remove_restricted_security("ABC")
        assert result is True
        assert "ABC" not in monitor.restricted_securities

    def test_is_restricted(self, monitor):
        """测试检查是否受限"""
        security = create_restricted_security(
            symbol="BLOCKED",
            restriction_type=RestrictionType.BLACKLIST,
            reason="禁止交易",
        )
        monitor.add_restricted_security(security)
        assert monitor.is_restricted("BLOCKED") is True
        assert monitor.is_restricted("AAPL") is False

    def test_is_restricted_with_exemption(self, monitor):
        """测试豁免账户"""
        security = create_restricted_security(
            symbol="EXEMPT",
            restriction_type=RestrictionType.BLACKLIST,
            reason="测试豁免",
        )
        security.approved_accounts = ["ACC001"]
        monitor.add_restricted_security(security)

        assert monitor.is_restricted("EXEMPT", "ACC001") is False
        assert monitor.is_restricted("EXEMPT", "ACC002") is True

    def test_requires_approval(self, monitor):
        """测试是否需要审批"""
        security = create_restricted_security(
            symbol="WATCHED",
            restriction_type=RestrictionType.WATCHLIST,
            reason="需要审批",
        )
        monitor.add_restricted_security(security)
        assert monitor.requires_approval("WATCHED") is True
        assert monitor.requires_approval("AAPL") is False

    def test_get_all_restricted_by_type(self, monitor):
        """测试按类型获取受限证券"""
        monitor.add_restricted_security(create_restricted_security(
            symbol="BL1", restriction_type=RestrictionType.BLACKLIST, reason="1"
        ))
        monitor.add_restricted_security(create_restricted_security(
            symbol="BL2", restriction_type=RestrictionType.BLACKLIST, reason="2"
        ))
        monitor.add_restricted_security(create_restricted_security(
            symbol="WL1", restriction_type=RestrictionType.WATCHLIST, reason="3"
        ))

        blacklist = monitor.get_all_restricted_securities(RestrictionType.BLACKLIST)
        assert len(blacklist) == 2


# ==================== 合规检查测试 ====================

class TestComplianceChecks:
    """合规检查测试"""

    def test_check_trade_passes(self, monitor, sample_trade):
        """测试正常交易通过检查"""
        results = monitor.check_trade(sample_trade)
        # 应该有多个检查结果
        assert len(results) > 0
        # 小额交易应该通过
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count > 0

    def test_check_position_limit_violation(self, monitor, position_provider):
        """测试持仓限制违规"""
        # 设置ADV为1000
        monitor.market_data_provider.set_adv("AAPL", 1000)

        # 创建会超限的交易 (尝试买入200，当前已有1000，ADV=1000，超过10%)
        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=200,
            price=150.0,
        )

        results = monitor.check_trade(trade)
        position_limit_results = [r for r in results if r.rule_type == RuleType.POSITION_LIMIT]

        # 应该有持仓限制违规
        assert any(not r.passed for r in position_limit_results)

    def test_check_concentration_violation(self, monitor, position_provider):
        """测试集中度违规"""
        # 当前持仓1000股，价格150，组合价值100万
        # 集中度 = 1000 * 150 / 1000000 = 15%
        # 如果买入更多，会超过20%

        # 买入500股，新持仓1500股
        # 新集中度 = 1500 * 150 / 1000000 = 22.5% > 20%
        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=500,
            price=150.0,
        )

        results = monitor.check_trade(trade)
        concentration_results = [r for r in results if r.rule_type == RuleType.CONCENTRATION]

        assert len(concentration_results) > 0
        assert any(not r.passed for r in concentration_results)

    def test_check_trading_frequency_violation(self, monitor):
        """测试交易频率违规"""
        # 同一分钟内超过10次交易
        for i in range(15):
            trade = create_trade_for_compliance(
                account_id="ACC001",
                symbol="AAPL",
                side="buy",
                quantity=10,
                price=150.0,
            )
            results = monitor.check_trade(trade)

        # 最后几次应该触发频率违规
        frequency_results = [r for r in results if r.rule_type == RuleType.TRADING_FREQUENCY]
        assert any(not r.passed for r in frequency_results)

    def test_check_wash_trading_detection(self, monitor):
        """测试洗售交易检测"""
        # 先买入
        buy_trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )
        monitor.check_trade(buy_trade)

        # 立即卖出相同数量，相似价格
        sell_trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            price=150.1,  # 价格非常接近
        )
        results = monitor.check_trade(sell_trade)

        wash_results = [r for r in results if r.rule_type == RuleType.WASH_TRADING]
        assert len(wash_results) > 0
        assert any(not r.passed for r in wash_results)

    def test_check_restricted_security_blacklist(self, monitor):
        """测试黑名单证券检查"""
        # 添加黑名单
        monitor.add_restricted_security(create_restricted_security(
            symbol="BLOCKED",
            restriction_type=RestrictionType.BLACKLIST,
            reason="禁止交易",
        ))

        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="BLOCKED",
            side="buy",
            quantity=100,
            price=50.0,
        )
        results = monitor.check_trade(trade)

        restricted_results = [r for r in results if r.rule_type == RuleType.RESTRICTED_SECURITY]
        assert len(restricted_results) > 0
        assert any(not r.passed for r in restricted_results)

    def test_check_restricted_security_watchlist(self, monitor):
        """测试观察名单证券检查"""
        monitor.add_restricted_security(create_restricted_security(
            symbol="WATCHED",
            restriction_type=RestrictionType.WATCHLIST,
            reason="需审批",
        ))

        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="WATCHED",
            side="buy",
            quantity=100,
            price=50.0,
        )
        results = monitor.check_trade(trade)

        restricted_results = [r for r in results if r.rule_type == RuleType.RESTRICTED_SECURITY]
        assert any(not r.passed for r in restricted_results)

    def test_check_order_size_violation(self, monitor):
        """测试订单大小违规"""
        # 添加订单大小规则
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="订单金额限制",
            description="限制订单金额",
            condition={"max_order_value": 10000},
            severity=RuleSeverity.VIOLATION,
        )
        monitor.add_rule(rule)

        # 创建超限订单
        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,  # 金额15000 > 10000
        )
        results = monitor.check_trade(trade)

        order_size_results = [r for r in results if r.rule_type == RuleType.ORDER_SIZE]
        assert any(not r.passed for r in order_size_results)


# ==================== 违规管理测试 ====================

class TestViolationManagement:
    """违规管理测试"""

    def test_violation_created_on_failure(self, monitor):
        """测试失败检查创建违规"""
        # 设置ADV为100，使小交易也会违规
        monitor.market_data_provider.set_adv("AAPL", 100)

        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )
        monitor.check_trade(trade)

        violations = monitor.get_violations()
        assert len(violations) > 0

    def test_get_violations_by_status(self, monitor):
        """测试按状态获取违规"""
        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        open_violations = monitor.get_violations(status=ViolationStatus.OPEN)
        assert len(open_violations) > 0

    def test_update_violation_status(self, monitor):
        """测试更新违规状态"""
        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        violations = monitor.get_violations()
        violation_id = violations[0].violation_id

        result = monitor.update_violation_status(
            violation_id, ViolationStatus.RESOLVED, "已处理"
        )
        assert result is True

        violation = monitor.get_violation(violation_id)
        assert violation.status == ViolationStatus.RESOLVED
        assert violation.resolved_at is not None

    def test_assign_violation(self, monitor):
        """测试分配违规处理人"""
        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        violations = monitor.get_violations()
        violation_id = violations[0].violation_id

        result = monitor.assign_violation(violation_id, "compliance_officer")
        assert result is True

        violation = monitor.get_violation(violation_id)
        assert violation.assigned_to == "compliance_officer"
        assert violation.status == ViolationStatus.UNDER_REVIEW

    def test_violation_callback(self, monitor):
        """测试违规回调"""
        callback_called = []

        def on_violation(violation):
            callback_called.append(violation)

        monitor.on_violation(on_violation)

        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        assert len(callback_called) > 0


# ==================== 审计日志测试 ====================

class TestAuditLog:
    """审计日志测试"""

    def test_audit_on_add_rule(self, monitor):
        """测试添加规则时记录审计"""
        initial_log_size = len(monitor.audit_log)

        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="测试规则",
            description="测试",
            condition={},
        )
        monitor.add_rule(rule, actor="admin")

        assert len(monitor.audit_log) > initial_log_size

        # 检查最后一条审计记录
        last_entry = monitor.audit_log[-1]
        assert last_entry.action == "add_rule"
        assert last_entry.actor == "admin"

    def test_audit_on_update_violation(self, monitor):
        """测试更新违规时记录审计"""
        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        violations = monitor.get_violations()
        violation_id = violations[0].violation_id

        initial_log_size = len(monitor.audit_log)
        monitor.update_violation_status(violation_id, ViolationStatus.RESOLVED, actor="officer")

        assert len(monitor.audit_log) > initial_log_size

    def test_get_audit_log_filtered(self, monitor):
        """测试筛选审计日志"""
        # 添加几条规则
        for i in range(3):
            rule = create_compliance_rule(
                rule_type=RuleType.ORDER_SIZE,
                name=f"规则{i}",
                description="测试",
                condition={},
            )
            monitor.add_rule(rule, actor="admin")

        entries = monitor.get_audit_log(target_type="rule")
        assert all(e.target_type == "rule" for e in entries)


# ==================== 报告生成测试 ====================

class TestReportGeneration:
    """报告生成测试"""

    def test_generate_daily_report(self, monitor):
        """测试生成日报"""
        # 创建一些交易和违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        for _ in range(5):
            trade = create_trade_for_compliance(
                account_id="ACC001", symbol="AAPL", side="buy", quantity=50, price=150.0
            )
            monitor.check_trade(trade)

        report = monitor.generate_report("daily")
        assert report.report_type == "daily"
        assert report.total_trades_checked >= 5

    def test_generate_weekly_report(self, monitor):
        """测试生成周报"""
        report = monitor.generate_report("weekly")
        assert report.report_type == "weekly"

    def test_report_violation_counts(self, monitor):
        """测试报告违规统计"""
        # 创建违规
        monitor.market_data_provider.set_adv("AAPL", 100)
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        monitor.check_trade(trade)

        report = monitor.generate_report("daily")
        assert report.violations_detected > 0

    def test_report_compliance_rate(self, monitor):
        """测试合规率计算"""
        # 创建一些正常交易
        for _ in range(5):
            trade = create_trade_for_compliance(
                account_id="ACC001", symbol="AAPL", side="buy", quantity=10, price=150.0
            )
            monitor.check_trade(trade)

        report = monitor.generate_report("daily")
        assert 0 <= report.compliance_rate <= 100

    def test_report_serialization(self, monitor):
        """测试报告序列化"""
        report = monitor.generate_report("daily")
        data = report.to_dict()

        assert "report_id" in data
        assert "report_type" in data
        assert "compliance_rate" in data


# ==================== 监控控制测试 ====================

class TestMonitoringControl:
    """监控控制测试"""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """测试启动停止监控"""
        await monitor.start_monitoring(check_interval=0.1)
        assert monitor.state == MonitorState.RUNNING

        await asyncio.sleep(0.2)

        await monitor.stop_monitoring()
        assert monitor.state == MonitorState.STOPPED

    def test_get_statistics(self, monitor):
        """测试获取统计信息"""
        # 执行一些检查
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=10, price=150.0
        )
        monitor.check_trade(trade)

        stats = monitor.get_statistics()
        assert "total_checks" in stats
        assert "total_rules" in stats
        assert stats["total_checks"] > 0

    def test_reset_statistics(self, monitor):
        """测试重置统计"""
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=10, price=150.0
        )
        monitor.check_trade(trade)

        monitor.reset_statistics()
        stats = monitor.get_statistics()
        assert stats["total_checks"] == 0


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_compliance_monitor(self):
        """测试创建合规监控引擎"""
        monitor = create_compliance_monitor()
        assert monitor is not None
        assert len(monitor.rules) > 0  # 有默认规则

    def test_create_compliance_rule(self):
        """测试创建合规规则"""
        rule = create_compliance_rule(
            rule_type=RuleType.CONCENTRATION,
            name="集中度限制",
            description="测试",
            condition={"max_pct": 0.2},
        )
        assert rule.rule_type == RuleType.CONCENTRATION

    def test_create_trade_for_compliance(self):
        """测试创建待检查交易"""
        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )
        assert trade.account_id == "ACC001"
        assert trade.trade_id is not None

    def test_create_restricted_security(self):
        """测试创建受限证券"""
        security = create_restricted_security(
            symbol="XYZ",
            restriction_type=RestrictionType.BLACKLIST,
            reason="测试",
        )
        assert security.symbol == "XYZ"

    def test_quick_compliance_check(self):
        """测试快速合规检查"""
        results = quick_compliance_check(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )
        assert len(results) > 0


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_trade_history(self, monitor):
        """测试空交易历史"""
        # 洗售检测在空历史时应该通过
        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        results = monitor.check_trade(trade)
        wash_results = [r for r in results if r.rule_type == RuleType.WASH_TRADING]
        assert all(r.passed for r in wash_results)

    def test_zero_adv(self, monitor):
        """测试ADV为0"""
        monitor.market_data_provider.set_adv("ZERO", 0)

        trade = create_trade_for_compliance(
            account_id="ACC001", symbol="ZERO", side="buy", quantity=100, price=100.0
        )
        # 不应该抛出异常
        results = monitor.check_trade(trade)
        assert len(results) > 0

    def test_zero_portfolio_value(self, monitor, position_provider):
        """测试组合价值为0"""
        position_provider.set_portfolio_value("EMPTY_ACC", 0)

        trade = create_trade_for_compliance(
            account_id="EMPTY_ACC", symbol="AAPL", side="buy", quantity=100, price=150.0
        )
        # 不应该抛出异常
        results = monitor.check_trade(trade)
        assert len(results) > 0

    def test_nonexistent_rule_update(self, monitor):
        """测试更新不存在的规则"""
        result = monitor.update_rule("nonexistent", {"enabled": False})
        assert result is False

    def test_nonexistent_violation_update(self, monitor):
        """测试更新不存在的违规"""
        result = monitor.update_violation_status("nonexistent", ViolationStatus.RESOLVED)
        assert result is False

    def test_rule_not_applicable(self, monitor):
        """测试规则不适用的情况"""
        # 添加只适用于特定账户的规则
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="特定账户规则",
            description="测试",
            condition={"max_order_value": 1000},
        )
        rule.applies_to_accounts = ["SPECIAL_ACC"]
        monitor.add_rule(rule)

        # 其他账户的交易不应该触发此规则
        trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,  # 金额15000 > 1000，但规则不适用
        )
        results = monitor.check_trade(trade)

        # 该规则不应该出现在结果中（因为不适用）
        matching_results = [r for r in results if r.rule_id == rule.rule_id]
        assert len(matching_results) == 0


# ==================== 序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    def test_config_to_dict(self, config):
        """测试配置序列化"""
        data = config.to_dict()
        assert "wash_trade_time_window_seconds" in data
        assert "max_trades_per_minute" in data

    def test_check_result_to_dict(self, monitor, sample_trade):
        """测试检查结果序列化"""
        results = monitor.check_trade(sample_trade)
        for result in results:
            data = result.to_dict()
            assert "passed" in data
            assert "rule_id" in data
            assert "rule_type" in data


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_compliance_workflow(self, monitor):
        """测试完整合规工作流"""
        # 1. 添加自定义规则
        rule = create_compliance_rule(
            rule_type=RuleType.ORDER_SIZE,
            name="大额订单限制",
            description="限制大额订单",
            condition={"max_order_value": 100000},
            severity=RuleSeverity.WARNING,
        )
        monitor.add_rule(rule)

        # 2. 添加受限证券
        security = create_restricted_security(
            symbol="RISKY",
            restriction_type=RestrictionType.WATCHLIST,
            reason="高风险",
        )
        monitor.add_restricted_security(security)

        # 3. 设置合适的ADV使小额交易能通过
        monitor.market_data_provider.set_adv("AAPL", 100000)  # 大ADV
        monitor.position_provider.set_position("ACC001", "AAPL", 0)  # 清空持仓

        # 4. 检查正常交易
        normal_trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
        )
        results1 = monitor.check_trade(normal_trade)
        # 检查是否有任何严重违规（允许INFO级别的通过）
        serious_failures = [r for r in results1 if not r.passed and r.violation and r.violation.severity in [RuleSeverity.VIOLATION, RuleSeverity.CRITICAL]]
        assert len(serious_failures) == 0

        # 5. 检查违规交易
        monitor.market_data_provider.set_adv("AAPL", 100)
        violation_trade = create_trade_for_compliance(
            account_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=200,
            price=150.0,
        )
        results2 = monitor.check_trade(violation_trade)
        violations = [r for r in results2 if not r.passed]
        assert len(violations) > 0

        # 6. 处理违规
        violation = monitor.get_violations()[0]
        monitor.assign_violation(violation.violation_id, "compliance_officer")
        monitor.update_violation_status(
            violation.violation_id,
            ViolationStatus.RESOLVED,
            "已审核通过"
        )

        # 7. 生成报告
        report = monitor.generate_report("daily")
        assert report.violations_detected > 0
        assert report.violations_resolved > 0

        # 8. 检查审计日志
        audit_entries = monitor.get_audit_log(limit=10)
        assert len(audit_entries) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
