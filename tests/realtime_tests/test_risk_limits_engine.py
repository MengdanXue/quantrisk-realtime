"""
风控限额引擎测试
Test Risk Limits Engine
"""

import asyncio
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.realtime.risk_limits_engine import (
    # 枚举
    LimitType,
    LimitLevel,
    LimitStatus,
    CheckResult,
    ApprovalStatus,
    EngineState,
    # 数据类
    RiskLimit,
    LimitCheckRequest,
    LimitCheckResult,
    LimitBreach,
    ApprovalRequest,
    LimitAlert,
    EngineConfig,
    # 接口
    PositionDataProvider,
    RiskMetricsProvider,
    # 模拟实现
    SimulatedPositionProvider,
    SimulatedRiskMetricsProvider,
    # 核心引擎
    RiskLimitsEngine,
    # 工厂函数
    create_risk_limits_engine,
    create_risk_limit,
    create_limit_check_request,
    quick_limit_check,
)


# ==================== 枚举测试 ====================

class TestEnums:
    """枚举测试"""

    def test_limit_type_values(self):
        """测试限额类型枚举值"""
        assert LimitType.GROSS_EXPOSURE.value == "gross_exposure"
        assert LimitType.NET_EXPOSURE.value == "net_exposure"
        assert LimitType.VAR_LIMIT.value == "var_limit"
        assert LimitType.DAILY_LOSS.value == "daily_loss"
        assert LimitType.ORDER_SIZE.value == "order_size"

    def test_limit_level_values(self):
        """测试限额级别枚举值"""
        assert LimitLevel.TRADER.value == "trader"
        assert LimitLevel.STRATEGY.value == "strategy"
        assert LimitLevel.PORTFOLIO.value == "portfolio"
        assert LimitLevel.ACCOUNT.value == "account"
        assert LimitLevel.DESK.value == "desk"
        assert LimitLevel.FIRM.value == "firm"

    def test_limit_status_values(self):
        """测试限额状态枚举值"""
        assert LimitStatus.NORMAL.value == "normal"
        assert LimitStatus.WARNING.value == "warning"
        assert LimitStatus.CRITICAL.value == "critical"
        assert LimitStatus.BREACHED.value == "breached"

    def test_check_result_values(self):
        """测试检查结果枚举值"""
        assert CheckResult.PASSED.value == "passed"
        assert CheckResult.SOFT_REJECT.value == "soft_reject"
        assert CheckResult.HARD_REJECT.value == "hard_reject"

    def test_approval_status_values(self):
        """测试审批状态枚举值"""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"


# ==================== 数据类测试 ====================

class TestRiskLimit:
    """风控限额数据类测试"""

    def test_create_limit(self):
        """测试创建限额"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
        )

        assert limit.limit_id == "L001"
        assert limit.limit_type == LimitType.GROSS_EXPOSURE
        assert limit.entity_id == "ACC001"
        assert limit.limit_value == Decimal("1000000")

    def test_usage_ratio(self):
        """测试使用率计算"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("500000"),
        )

        assert limit.usage_ratio == Decimal("0.5")

    def test_usage_ratio_zero_limit(self):
        """测试零限额的使用率"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("0"),
            current_usage=Decimal("100"),
        )

        assert limit.usage_ratio == Decimal("0")

    def test_available_calculation(self):
        """测试可用额度计算"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("400000"),
        )

        assert limit.available == Decimal("600000")

    def test_status_normal(self):
        """测试正常状态"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("500000"),  # 50%
        )

        assert limit.status == LimitStatus.NORMAL

    def test_status_warning(self):
        """测试预警状态"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("850000"),  # 85%
            warning_threshold=Decimal("0.80"),
        )

        assert limit.status == LimitStatus.WARNING

    def test_status_critical(self):
        """测试严重状态"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("950000"),  # 95%
            critical_threshold=Decimal("0.90"),
        )

        assert limit.status == LimitStatus.CRITICAL

    def test_status_breached(self):
        """测试突破状态"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("1100000"),  # 110%
        )

        assert limit.status == LimitStatus.BREACHED

    def test_status_expired(self):
        """测试过期状态"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            expiry_date=date.today() - timedelta(days=1),
        )

        assert limit.status == LimitStatus.EXPIRED

    def test_to_dict(self):
        """测试字典转换"""
        limit = RiskLimit(
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            limit_level=LimitLevel.ACCOUNT,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            current_usage=Decimal("500000"),
        )

        d = limit.to_dict()
        assert d["limit_id"] == "L001"
        assert d["limit_type"] == "gross_exposure"
        assert d["limit_level"] == "account"
        assert d["limit_value"] == 1000000.0
        assert d["usage_ratio"] == 0.5


class TestLimitCheckRequest:
    """限额检查请求测试"""

    def test_create_request(self):
        """测试创建请求"""
        request = LimitCheckRequest(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )

        assert request.order_id == "ORD001"
        assert request.symbol == "AAPL"
        assert request.side == "buy"
        assert request.quantity == Decimal("100")
        assert request.price == Decimal("150.00")

    def test_notional_calculation(self):
        """测试名义金额计算"""
        request = LimitCheckRequest(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )

        assert request.notional == Decimal("15000.00")

    def test_to_dict(self):
        """测试字典转换"""
        request = LimitCheckRequest(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            trader_id="TR001",
        )

        d = request.to_dict()
        assert d["order_id"] == "ORD001"
        assert d["notional"] == 15000.00
        assert d["trader_id"] == "TR001"


class TestLimitBreach:
    """限额突破记录测试"""

    def test_create_breach(self):
        """测试创建突破记录"""
        breach = LimitBreach(
            breach_id="B001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            actual_value=Decimal("1200000"),
            breach_ratio=Decimal("1.2"),
            order_id="ORD001",
        )

        assert breach.breach_id == "B001"
        assert breach.breach_ratio == Decimal("1.2")
        assert breach.resolved is False

    def test_to_dict(self):
        """测试字典转换"""
        breach = LimitBreach(
            breach_id="B001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            limit_value=Decimal("1000000"),
            actual_value=Decimal("1200000"),
            breach_ratio=Decimal("1.2"),
            order_id="ORD001",
        )

        d = breach.to_dict()
        assert d["breach_id"] == "B001"
        assert d["limit_type"] == "gross_exposure"
        assert d["breach_ratio"] == 1.2


class TestApprovalRequest:
    """审批请求测试"""

    def test_create_approval(self):
        """测试创建审批请求"""
        approval = ApprovalRequest(
            approval_id="A001",
            order_id="ORD001",
            entity_id="ACC001",
            violated_limits=["L001", "L002"],
            requested_by="TR001",
            reason="Order exceeds soft limits",
        )

        assert approval.approval_id == "A001"
        assert approval.status == ApprovalStatus.PENDING
        assert len(approval.violated_limits) == 2

    def test_to_dict(self):
        """测试字典转换"""
        approval = ApprovalRequest(
            approval_id="A001",
            order_id="ORD001",
            entity_id="ACC001",
            violated_limits=["L001"],
            requested_by="TR001",
            reason="Test",
        )

        d = approval.to_dict()
        assert d["approval_id"] == "A001"
        assert d["status"] == "pending"


class TestLimitAlert:
    """限额预警测试"""

    def test_create_alert(self):
        """测试创建预警"""
        alert = LimitAlert(
            alert_id="AL001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            alert_level="warning",
            usage_ratio=Decimal("0.85"),
            message="Limit at 85%",
        )

        assert alert.alert_id == "AL001"
        assert alert.acknowledged is False

    def test_to_dict(self):
        """测试字典转换"""
        alert = LimitAlert(
            alert_id="AL001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            alert_level="warning",
            usage_ratio=Decimal("0.85"),
            message="Test",
        )

        d = alert.to_dict()
        assert d["alert_level"] == "warning"
        assert d["usage_ratio"] == 0.85


# ==================== 模拟提供者测试 ====================

class TestSimulatedProviders:
    """模拟提供者测试"""

    @pytest.mark.asyncio
    async def test_position_provider(self):
        """测试模拟持仓提供者"""
        provider = SimulatedPositionProvider()

        # 初始为空
        positions = await provider.get_positions("ACC001")
        assert positions == []

        # 设置敞口
        provider.set_exposure("ACC001", Decimal("100000"), Decimal("50000"))

        gross = await provider.get_gross_exposure("ACC001")
        net = await provider.get_net_exposure("ACC001")

        assert gross == Decimal("100000")
        assert net == Decimal("50000")

    @pytest.mark.asyncio
    async def test_risk_metrics_provider(self):
        """测试模拟风险指标提供者"""
        provider = SimulatedRiskMetricsProvider()

        # 设置指标
        provider.set_metrics(
            "ACC001",
            var=Decimal("50000"),
            cvar=Decimal("75000"),
            drawdown=Decimal("0.05"),
        )

        var = await provider.get_var("ACC001")
        cvar = await provider.get_cvar("ACC001")
        drawdown = await provider.get_drawdown("ACC001")

        assert var == Decimal("50000")
        assert cvar == Decimal("75000")
        assert drawdown == Decimal("0.05")


# ==================== 引擎核心功能测试 ====================

class TestRiskLimitsEngine:
    """风控限额引擎测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    def test_add_limit(self):
        """测试添加限额"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )

        limit_id = self.engine.add_limit(limit)
        assert limit_id == limit.limit_id
        assert len(self.engine.limits) == 1

    def test_get_limit(self):
        """测试获取限额"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        retrieved = self.engine.get_limit(limit.limit_id)
        assert retrieved is not None
        assert retrieved.entity_id == "ACC001"

    def test_remove_limit(self):
        """测试移除限额"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        result = self.engine.remove_limit(limit.limit_id)
        assert result is True
        assert len(self.engine.limits) == 0

    def test_remove_nonexistent_limit(self):
        """测试移除不存在的限额"""
        result = self.engine.remove_limit("nonexistent")
        assert result is False

    def test_get_limits_by_entity(self):
        """测试按实体获取限额"""
        limit1 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        limit2 = create_risk_limit(
            limit_type="daily_loss",
            entity_id="ACC001",
            limit_value=50000,
        )
        limit3 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC002",
            limit_value=500000,
        )

        self.engine.add_limit(limit1)
        self.engine.add_limit(limit2)
        self.engine.add_limit(limit3)

        acc001_limits = self.engine.get_limits_by_entity("ACC001")
        assert len(acc001_limits) == 2

    def test_get_limits_by_type(self):
        """测试按类型获取限额"""
        limit1 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        limit2 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC002",
            limit_value=500000,
        )

        self.engine.add_limit(limit1)
        self.engine.add_limit(limit2)

        gross_limits = self.engine.get_limits_by_type(LimitType.GROSS_EXPOSURE)
        assert len(gross_limits) == 2

    def test_update_limit(self):
        """测试更新限额"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        result = self.engine.update_limit(
            limit.limit_id,
            limit_value=Decimal("1500000"),
            warning_threshold=Decimal("0.75"),
        )

        assert result is True
        updated = self.engine.get_limit(limit.limit_id)
        assert updated.limit_value == Decimal("1500000")
        assert updated.warning_threshold == Decimal("0.75")

    def test_update_usage(self):
        """测试更新使用量"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        result = self.engine.update_usage(limit.limit_id, Decimal("500000"))

        assert result is True
        updated = self.engine.get_limit(limit.limit_id)
        assert updated.current_usage == Decimal("500000")


class TestPreTradeCheck:
    """Pre-Trade 检查测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    @pytest.mark.asyncio
    async def test_check_order_passed(self):
        """测试订单检查通过"""
        # 添加总敞口限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        # 创建检查请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # notional = 15000
        )

        result = await self.engine.check_order(request)

        assert result.result == CheckResult.PASSED
        assert len(result.violated_limits) == 0

    @pytest.mark.asyncio
    async def test_check_order_hard_reject(self):
        """测试订单硬拒绝"""
        # 添加小限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=10000,
            is_hard_limit=True,
        )
        self.engine.add_limit(limit)

        # 创建超限请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # notional = 15000 > 10000
        )

        result = await self.engine.check_order(request)

        assert result.result == CheckResult.HARD_REJECT
        assert len(result.violated_limits) > 0

    @pytest.mark.asyncio
    async def test_check_order_soft_reject(self):
        """测试订单软拒绝"""
        # 添加软限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=10000,
            is_hard_limit=False,  # 软限额
        )
        self.engine.add_limit(limit)

        # 创建超限请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # notional = 15000 > 10000
        )

        result = await self.engine.check_order(request)

        assert result.result == CheckResult.SOFT_REJECT
        assert result.approval_required is True
        assert result.approval_id is not None

    @pytest.mark.asyncio
    async def test_check_order_with_warning(self):
        """测试订单检查带警告"""
        # 添加限额，设置已使用75%
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=100000,
            warning_threshold=0.80,
            critical_threshold=0.95,  # 设置更高的critical阈值
        )
        limit.current_usage = Decimal("75000")
        self.engine.add_limit(limit)

        # 创建请求，使总使用达到90% (在warning和critical之间)
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # notional = 15000
        )

        result = await self.engine.check_order(request)

        assert result.result == CheckResult.PASSED
        # 检查已检查的限额包含warning状态
        assert len(result.checked_limits) > 0
        assert result.checked_limits[0]["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_order_size_limit(self):
        """测试单笔订单大小限额"""
        # 添加单笔订单限额
        limit = create_risk_limit(
            limit_type="order_size",
            entity_id="ACC001",
            limit_value=10000,
            is_hard_limit=True,
        )
        self.engine.add_limit(limit)

        # 创建超限请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # notional = 15000 > 10000
        )

        result = await self.engine.check_order(request)
        assert result.result == CheckResult.HARD_REJECT

    @pytest.mark.asyncio
    async def test_check_daily_trades_limit(self):
        """测试日交易次数限额"""
        # 添加日交易限额 - 限制为10次
        limit = create_risk_limit(
            limit_type="daily_trades",
            entity_id="ACC001",
            limit_value=10,
            is_hard_limit=True,
        )
        self.engine.add_limit(limit)

        # 设置已交易8次
        self.engine.update_daily_counter("ACC001", trades=8)

        # 再交易一次应该可以 (8+1=9 < 10)
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.00,
        )

        result = await self.engine.check_order(request)
        assert result.result == CheckResult.PASSED

        # 更新到9次
        self.engine.update_daily_counter("ACC001", trades=1)

        # 再交易应该被拒绝 (9+1=10 >= 10)
        request2 = create_limit_check_request(
            order_id="ORD002",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.00,
        )

        result2 = await self.engine.check_order(request2)
        assert result2.result == CheckResult.HARD_REJECT

    @pytest.mark.asyncio
    async def test_cascade_limit_check(self):
        """测试级联限额检查"""
        # 添加账户级限额
        account_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
            limit_level="account",
        )

        # 添加交易员级限额
        trader_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="TR001",
            limit_value=500000,
            limit_level="trader",
        )

        self.engine.add_limit(account_limit)
        self.engine.add_limit(trader_limit)

        # 创建请求，包含trader_id
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
            trader_id="TR001",
        )

        result = await self.engine.check_order(request)

        # 应该检查了两个限额
        assert len(result.checked_limits) >= 2


class TestApprovalManagement:
    """审批管理测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    @pytest.mark.asyncio
    async def test_approve_request(self):
        """测试批准请求"""
        # 添加软限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=10000,
            is_hard_limit=False,
        )
        self.engine.add_limit(limit)

        # 创建超限请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )

        result = await self.engine.check_order(request)
        approval_id = result.approval_id

        # 批准
        approved = self.engine.approve_request(approval_id, "MANAGER001", "Override approved")

        assert approved is True
        assert self.engine.approval_requests[approval_id].status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_reject_request(self):
        """测试拒绝请求"""
        # 添加软限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=10000,
            is_hard_limit=False,
        )
        self.engine.add_limit(limit)

        # 创建超限请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )

        result = await self.engine.check_order(request)
        approval_id = result.approval_id

        # 拒绝
        rejected = self.engine.reject_request(approval_id, "MANAGER001", "Exceeds risk tolerance")

        assert rejected is True
        assert self.engine.approval_requests[approval_id].status == ApprovalStatus.REJECTED

    def test_get_pending_approvals(self):
        """测试获取待审批列表"""
        # 手动添加审批请求
        approval = ApprovalRequest(
            approval_id="A001",
            order_id="ORD001",
            entity_id="ACC001",
            violated_limits=["L001"],
            requested_by="TR001",
            reason="Test",
        )
        self.engine.approval_requests["A001"] = approval

        pending = self.engine.get_pending_approvals()
        assert len(pending) == 1

    def test_approve_nonexistent_request(self):
        """测试批准不存在的请求"""
        result = self.engine.approve_request("nonexistent", "MANAGER001")
        assert result is False


class TestAlertManagement:
    """告警管理测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    def test_alert_on_warning(self):
        """测试预警状态告警"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
            warning_threshold=0.80,
        )
        self.engine.add_limit(limit)

        # 更新使用量触发预警
        self.engine.update_usage(limit.limit_id, Decimal("850000"))

        # 应该生成告警
        assert len(self.engine.alerts) == 1
        assert self.engine.alerts[0].alert_level == "warning"

    def test_alert_on_breach(self):
        """测试突破状态告警"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        # 更新使用量触发突破
        self.engine.update_usage(limit.limit_id, Decimal("1100000"))

        # 应该有告警和突破记录
        assert len(self.engine.alerts) == 1
        assert len(self.engine.breaches) == 1

    def test_acknowledge_alert(self):
        """测试确认告警"""
        # 添加告警
        alert = LimitAlert(
            alert_id="AL001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            alert_level="warning",
            usage_ratio=Decimal("0.85"),
            message="Test alert",
        )
        self.engine.alerts.append(alert)

        # 确认告警
        result = self.engine.acknowledge_alert("AL001", "MANAGER001")

        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "MANAGER001"

    def test_get_unacknowledged_alerts(self):
        """测试获取未确认告警"""
        alert1 = LimitAlert(
            alert_id="AL001",
            limit_id="L001",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            alert_level="warning",
            usage_ratio=Decimal("0.85"),
            message="Alert 1",
        )
        alert2 = LimitAlert(
            alert_id="AL002",
            limit_id="L002",
            limit_type=LimitType.GROSS_EXPOSURE,
            entity_id="ACC001",
            alert_level="critical",
            usage_ratio=Decimal("0.95"),
            message="Alert 2",
            acknowledged=True,
        )

        self.engine.alerts.extend([alert1, alert2])

        unacked = self.engine.get_unacknowledged_alerts()
        assert len(unacked) == 1
        assert unacked[0].alert_id == "AL001"


class TestDailyCounters:
    """日内计数器测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    def test_update_daily_counter(self):
        """测试更新日内计数器"""
        self.engine.update_daily_counter(
            "ACC001",
            trades=5,
            turnover=Decimal("100000"),
            loss=Decimal("1000"),
        )

        counter = self.engine._get_daily_counter("ACC001")

        assert counter["trades"] == 5
        assert counter["turnover"] == Decimal("100000")
        assert counter["loss"] == Decimal("1000")

    def test_counter_accumulation(self):
        """测试计数器累加"""
        self.engine.update_daily_counter("ACC001", trades=3)
        self.engine.update_daily_counter("ACC001", trades=2)

        counter = self.engine._get_daily_counter("ACC001")
        assert counter["trades"] == 5


class TestRealTimeMonitoring:
    """实时监控测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动停止"""
        await self.engine.start()
        assert self.engine.state == EngineState.RUNNING

        await self.engine.stop()
        assert self.engine.state == EngineState.IDLE

    @pytest.mark.asyncio
    async def test_monitoring_updates_limits(self):
        """测试监控更新限额"""
        # 使用模拟提供者
        position_provider = SimulatedPositionProvider()
        position_provider.set_exposure("ACC001", Decimal("500000"), Decimal("200000"))

        self.engine.position_provider = position_provider

        # 添加限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)

        # 手动触发更新
        await self.engine._update_limit_usage(limit)

        updated = self.engine.get_limit(limit.limit_id)
        assert updated.current_usage == Decimal("500000")


class TestStatistics:
    """统计功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    def test_get_statistics(self):
        """测试获取统计"""
        # 添加一些限额
        limit1 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        limit2 = create_risk_limit(
            limit_type="daily_loss",
            entity_id="ACC001",
            limit_value=50000,
        )

        self.engine.add_limit(limit1)
        self.engine.add_limit(limit2)

        stats = self.engine.get_statistics()

        assert stats["total_limits"] == 2
        assert "limits_by_status" in stats

    def test_get_limit_summary(self):
        """测试获取限额摘要"""
        # 添加不同状态的限额
        limit1 = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        limit2 = create_risk_limit(
            limit_type="daily_loss",
            entity_id="ACC001",
            limit_value=50000,
        )
        limit2.current_usage = Decimal("45000")  # 90% - critical

        self.engine.add_limit(limit1)
        self.engine.add_limit(limit2)

        summary = self.engine.get_limit_summary("ACC001")

        assert summary["entity_id"] == "ACC001"
        assert summary["total_limits"] == 2


class TestCallbacks:
    """回调功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    def test_on_alert_callback(self):
        """测试告警回调"""
        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        self.engine.on_alert(on_alert)

        # 添加限额并触发告警
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
            warning_threshold=0.80,
        )
        self.engine.add_limit(limit)
        self.engine.update_usage(limit.limit_id, Decimal("850000"))

        assert len(alerts_received) == 1

    def test_on_breach_callback(self):
        """测试突破回调"""
        breaches_received = []

        def on_breach(breach):
            breaches_received.append(breach)

        self.engine.on_breach(on_breach)

        # 添加限额并触发突破
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
        )
        self.engine.add_limit(limit)
        self.engine.update_usage(limit.limit_id, Decimal("1100000"))

        assert len(breaches_received) == 1


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_risk_limits_engine(self):
        """测试创建引擎"""
        engine = create_risk_limits_engine(
            check_interval=30.0,
            enable_pre_trade=True,
            enable_monitoring=True,
        )

        assert engine is not None
        assert engine.config.check_interval == 30.0

    def test_create_risk_limit(self):
        """测试创建限额"""
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
            warning_threshold=0.75,
            is_hard_limit=True,
        )

        assert limit.limit_type == LimitType.GROSS_EXPOSURE
        assert limit.entity_id == "ACC001"
        assert limit.limit_value == Decimal("1000000")
        assert limit.warning_threshold == Decimal("0.75")

    def test_create_limit_check_request(self):
        """测试创建检查请求"""
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )

        assert request.order_id == "ORD001"
        assert request.notional == Decimal("15000.00")

    @pytest.mark.asyncio
    async def test_quick_limit_check(self):
        """测试快速限额检查"""
        engine = create_risk_limits_engine()

        # 添加限额
        limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="default",
            limit_value=1000000,
        )
        engine.add_limit(limit)

        result = await quick_limit_check(
            engine=engine,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )

        assert result["result"] == "passed"


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = RiskLimitsEngine()

    @pytest.mark.asyncio
    async def test_check_with_no_limits(self):
        """测试无限额时的检查"""
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )

        result = await self.engine.check_order(request)

        assert result.result == CheckResult.PASSED
        assert len(result.checked_limits) == 0

    def test_get_nonexistent_limit(self):
        """测试获取不存在的限额"""
        result = self.engine.get_limit("nonexistent")
        assert result is None

    def test_update_nonexistent_limit(self):
        """测试更新不存在的限额"""
        result = self.engine.update_limit("nonexistent", limit_value=Decimal("1000000"))
        assert result is False

    def test_update_usage_nonexistent_limit(self):
        """测试更新不存在限额的使用量"""
        result = self.engine.update_usage("nonexistent", Decimal("500000"))
        assert result is False

    def test_acknowledge_nonexistent_alert(self):
        """测试确认不存在的告警"""
        result = self.engine.acknowledge_alert("nonexistent", "MANAGER001")
        assert result is False

    def test_get_breaches_empty(self):
        """测试获取空突破记录"""
        breaches = self.engine.get_breaches()
        assert breaches == []

    def test_get_alerts_empty(self):
        """测试获取空告警"""
        alerts = self.engine.get_alerts()
        assert alerts == []


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建引擎
        engine = create_risk_limits_engine()

        # 2. 设置回调
        alerts = []
        approvals = []

        engine.on_alert(lambda a: alerts.append(a))
        engine.on_approval_required(lambda a: approvals.append(a))

        # 3. 添加限额
        hard_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=100000,
            is_hard_limit=True,
        )
        soft_limit = create_risk_limit(
            limit_type="order_size",
            entity_id="ACC001",
            limit_value=20000,
            is_hard_limit=False,
        )

        engine.add_limit(hard_limit)
        engine.add_limit(soft_limit)

        # 4. 通过的订单
        request1 = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,  # 15000
        )
        result1 = await engine.check_order(request1)
        assert result1.result == CheckResult.PASSED

        # 5. 需要审批的订单（超过软限额）
        request2 = create_limit_check_request(
            order_id="ORD002",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=200,
            price=150.00,  # 30000 > 20000
        )
        result2 = await engine.check_order(request2)
        assert result2.result == CheckResult.SOFT_REJECT
        assert len(approvals) == 1

        # 6. 批准审批
        engine.approve_request(result2.approval_id, "MANAGER001")

        # 7. 被拒绝的订单（超过硬限额）
        request3 = create_limit_check_request(
            order_id="ORD003",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=1000,
            price=150.00,  # 150000 > 100000
        )
        result3 = await engine.check_order(request3)
        assert result3.result == CheckResult.HARD_REJECT

        # 8. 验证统计
        stats = engine.get_statistics()
        assert stats["total_checks"] == 3
        assert stats["passed_checks"] == 1
        assert stats["rejected_checks"] >= 1

    @pytest.mark.asyncio
    async def test_multi_level_limits(self):
        """测试多层级限额"""
        engine = create_risk_limits_engine()

        # 公司级限额
        firm_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="FIRM001",
            limit_value=10000000,
            limit_level="firm",
        )

        # 账户级限额
        account_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="ACC001",
            limit_value=1000000,
            limit_level="account",
        )

        # 交易员级限额
        trader_limit = create_risk_limit(
            limit_type="gross_exposure",
            entity_id="TR001",
            limit_value=500000,
            limit_level="trader",
        )

        engine.add_limit(firm_limit)
        engine.add_limit(account_limit)
        engine.add_limit(trader_limit)

        # 创建带完整层级信息的请求
        request = create_limit_check_request(
            order_id="ORD001",
            entity_id="ACC001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
            trader_id="TR001",
            account_id="ACC001",
        )

        result = await engine.check_order(request)

        # 应该检查了多个层级的限额
        assert result.result == CheckResult.PASSED
        assert len(result.checked_limits) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
