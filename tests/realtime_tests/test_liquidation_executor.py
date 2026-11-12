"""
强平执行器单元测试
Liquidation Executor Unit Tests

测试覆盖:
- 执行器初始化
- 模拟执行(dry run)
- 真实API调用(mock)
- 重试机制
- 并发控制
- 错误处理
- 回调函数
- 状态跟踪

作者: TraderX Team
创建: 2025-12-23
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any
import aiohttp

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.liquidation_executor import (
    LiquidationExecutor,
    LiquidationExecutionResult,
    PositionCloseResult,
    ExecutorConfig,
    ExecutionStatus,
    CloseOrderType,
    create_liquidation_executor,
    create_liquidation_callback,
)


# ============ 测试用数据类 ============

@dataclass
class MockLiquidationOrder:
    """模拟强平订单"""
    order_id: str = "test-order-001"
    account_id: str = "test-account"
    reason: str = "margin_call"
    trigger_margin_ratio: float = 0.96
    target_margin_ratio: float = 0.50
    positions_to_close: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.positions_to_close:
            self.positions_to_close = [
                {
                    'position_id': 1001,
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'side': 'sell',
                    'current_price': 150.0,
                    'margin_to_release': 15000.0,
                    'unrealized_pnl': 500.0,
                },
                {
                    'position_id': 1002,
                    'symbol': 'GOOGL',
                    'quantity': 50,
                    'side': 'sell',
                    'current_price': 2800.0,
                    'margin_to_release': 70000.0,
                    'unrealized_pnl': -1200.0,
                },
            ]


# ============ Fixtures ============

@pytest.fixture
def executor_config():
    """执行器配置"""
    return ExecutorConfig(
        java_api_base_url="http://localhost:8080",
        auth_token="Bearer test-token",
        dry_run=True,
        max_retries=2,
        retry_delay_seconds=0.1,
        max_concurrent_closes=3,
    )


@pytest.fixture
def executor(executor_config):
    """创建执行器"""
    return LiquidationExecutor(executor_config)


@pytest.fixture
def mock_order():
    """模拟强平订单"""
    return MockLiquidationOrder()


@pytest.fixture
def single_position_order():
    """单持仓订单"""
    return MockLiquidationOrder(
        positions_to_close=[
            {
                'position_id': 1001,
                'symbol': 'AAPL',
                'quantity': 100,
                'current_price': 150.0,
                'margin_to_release': 15000.0,
                'unrealized_pnl': 500.0,
            }
        ]
    )


@pytest.fixture
def empty_order():
    """空订单 - 不使用__post_init__填充默认值"""
    @dataclass
    class EmptyOrder:
        order_id: str = "test-empty-order"
        account_id: str = "test-account"
        reason: str = "margin_call"
        positions_to_close: List[Dict[str, Any]] = field(default_factory=list)

    return EmptyOrder()


# ============ 基础测试 ============

class TestExecutorConfig:
    """测试执行器配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ExecutorConfig()

        assert config.java_api_base_url == "http://localhost:8080"
        assert config.auth_token is None
        assert config.dry_run is False
        assert config.max_retries == 3
        assert config.max_concurrent_closes == 5

    def test_custom_config(self):
        """测试自定义配置"""
        config = ExecutorConfig(
            java_api_base_url="http://api.example.com",
            auth_token="Bearer xxx",
            dry_run=True,
            max_retries=5,
        )

        assert config.java_api_base_url == "http://api.example.com"
        assert config.auth_token == "Bearer xxx"
        assert config.dry_run is True
        assert config.max_retries == 5


class TestExecutorInitialization:
    """测试执行器初始化"""

    def test_basic_init(self):
        """测试基本初始化"""
        executor = LiquidationExecutor()

        assert executor.config is not None
        assert executor._session is None
        assert executor._execution_count == 0

    def test_init_with_config(self, executor_config):
        """测试带配置初始化"""
        executor = LiquidationExecutor(executor_config)

        assert executor.config.dry_run is True
        assert executor.config.max_retries == 2

    def test_init_with_callbacks(self):
        """测试带回调函数初始化"""
        on_closed = AsyncMock()
        on_complete = AsyncMock()

        executor = LiquidationExecutor(
            on_position_closed=on_closed,
            on_execution_complete=on_complete,
        )

        assert executor.on_position_closed == on_closed
        assert executor.on_execution_complete == on_complete


# ============ Dry Run 测试 ============

class TestDryRunExecution:
    """测试模拟执行"""

    @pytest.mark.asyncio
    async def test_dry_run_basic(self, executor, mock_order):
        """测试基本模拟执行"""
        result = await executor.execute(mock_order)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.successful_closes == 2
        assert result.failed_closes == 0
        assert len(result.position_results) == 2

    @pytest.mark.asyncio
    async def test_dry_run_single_position(self, executor, single_position_order):
        """测试单持仓模拟执行"""
        result = await executor.execute(single_position_order)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.successful_closes == 1
        assert result.total_positions == 1

    @pytest.mark.asyncio
    async def test_dry_run_empty_order(self, executor, empty_order):
        """测试空订单模拟执行"""
        result = await executor.execute(empty_order)

        assert result.status == ExecutionStatus.FAILED
        assert "No positions to close" in result.errors

    @pytest.mark.asyncio
    async def test_dry_run_pnl_calculation(self, executor, mock_order):
        """测试模拟执行的盈亏计算"""
        result = await executor.execute(mock_order)

        # 500 + (-1200) = -700
        expected_pnl = 500.0 + (-1200.0)
        assert result.total_realized_pnl == pytest.approx(expected_pnl)

    @pytest.mark.asyncio
    async def test_dry_run_margin_released(self, executor, mock_order):
        """测试模拟执行的保证金释放"""
        result = await executor.execute(mock_order)

        # 15000 + 70000 = 85000
        expected_margin = 15000.0 + 70000.0
        assert result.margin_released == pytest.approx(expected_margin)


# ============ 验证测试 ============

class TestOrderValidation:
    """测试订单验证"""

    @pytest.mark.asyncio
    async def test_validate_empty_positions(self, executor, empty_order):
        """测试空持仓列表验证"""
        result = await executor.execute(empty_order)

        assert result.status == ExecutionStatus.FAILED
        assert any("No positions" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_missing_fields(self, executor):
        """测试缺少必要字段"""
        order = MockLiquidationOrder(
            positions_to_close=[
                {'quantity': 100},  # 缺少 position_id 和 symbol
            ]
        )

        result = await executor.execute(order)
        assert result.status == ExecutionStatus.FAILED


# ============ 结果数据结构测试 ============

class TestPositionCloseResult:
    """测试持仓平仓结果"""

    def test_basic_result(self):
        """测试基本结果"""
        result = PositionCloseResult(
            position_id=1001,
            symbol="AAPL",
            success=True,
            closed_quantity=100,
            close_price=150.0,
        )

        assert result.success is True
        assert result.symbol == "AAPL"

    def test_to_dict(self):
        """测试转换为字典"""
        result = PositionCloseResult(
            position_id=1001,
            symbol="AAPL",
            success=True,
        )

        d = result.to_dict()
        assert d['position_id'] == 1001
        assert d['symbol'] == "AAPL"
        assert d['success'] is True


class TestLiquidationExecutionResult:
    """测试强平执行结果"""

    def test_success_property(self):
        """测试成功属性"""
        result = LiquidationExecutionResult(order_id="test")
        result.status = ExecutionStatus.COMPLETED
        result.successful_closes = 2
        result.failed_closes = 0

        assert result.success is True

    def test_partial_success_property(self):
        """测试部分成功属性"""
        result = LiquidationExecutionResult(order_id="test")
        result.successful_closes = 1
        result.failed_closes = 1

        assert result.partial_success is True

    def test_add_audit_log(self):
        """测试添加审计日志"""
        result = LiquidationExecutionResult(order_id="test")
        result.add_audit_log("test_action", {"key": "value"})

        assert len(result.audit_log) == 1
        assert result.audit_log[0]['action'] == "test_action"

    def test_to_dict(self):
        """测试转换为字典"""
        result = LiquidationExecutionResult(order_id="test-123")
        result.status = ExecutionStatus.COMPLETED

        d = result.to_dict()
        assert d['order_id'] == "test-123"
        assert d['status'] == "completed"


# ============ 回调测试 ============

class TestCallbacks:
    """测试回调函数"""

    @pytest.mark.asyncio
    async def test_position_closed_callback(self, executor_config, mock_order):
        """测试单持仓平仓回调"""
        callback_results = []

        async def on_closed(result):
            callback_results.append(result)

        executor = LiquidationExecutor(
            executor_config,
            on_position_closed=on_closed,
        )

        await executor.execute(mock_order)

        assert len(callback_results) == 2
        assert all(r.success for r in callback_results)

    @pytest.mark.asyncio
    async def test_execution_complete_callback(self, executor_config, mock_order):
        """测试执行完成回调"""
        complete_result = None

        async def on_complete(result):
            nonlocal complete_result
            complete_result = result

        executor = LiquidationExecutor(
            executor_config,
            on_execution_complete=on_complete,
        )

        await executor.execute(mock_order)

        assert complete_result is not None
        assert complete_result.status == ExecutionStatus.COMPLETED


# ============ 统计测试 ============

class TestStatistics:
    """测试统计信息"""

    @pytest.mark.asyncio
    async def test_execution_count(self, executor, mock_order):
        """测试执行计数"""
        assert executor._execution_count == 0

        await executor.execute(mock_order)
        assert executor._execution_count == 1

        await executor.execute(mock_order)
        assert executor._execution_count == 2

    @pytest.mark.asyncio
    async def test_total_positions_closed(self, executor, mock_order):
        """测试总平仓数"""
        await executor.execute(mock_order)

        assert executor._total_positions_closed == 2

    @pytest.mark.asyncio
    async def test_get_statistics(self, executor, mock_order):
        """测试获取统计信息"""
        await executor.execute(mock_order)

        stats = executor.get_statistics()

        assert stats['execution_count'] == 1
        assert stats['total_positions_closed'] == 2
        assert 'config' in stats


# ============ 工厂函数测试 ============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_liquidation_executor(self):
        """测试创建执行器"""
        executor = create_liquidation_executor(
            java_api_url="http://api.example.com",
            auth_token="Bearer token",
            dry_run=True,
        )

        assert executor.config.java_api_base_url == "http://api.example.com"
        assert executor.config.auth_token == "Bearer token"
        assert executor.config.dry_run is True

    @pytest.mark.asyncio
    async def test_create_liquidation_callback(self, executor, mock_order):
        """测试创建回调函数"""
        callback = create_liquidation_callback(executor)

        result = await callback(mock_order)

        assert result['success'] is True
        assert result['successful_closes'] == 2
        assert result['failed_closes'] == 0


# ============ API调用测试(Mock) ============

class TestApiCalls:
    """测试API调用(使用Mock)"""

    @pytest.mark.asyncio
    async def test_successful_api_call(self, single_position_order):
        """测试成功的API调用"""
        config = ExecutorConfig(
            java_api_base_url="http://localhost:8080",
            dry_run=False,
        )
        executor = LiquidationExecutor(config)

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'code': 200,
            'message': '平仓成功',
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        with patch.object(executor, '_get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await executor.execute(single_position_order)

        assert result.successful_closes == 1
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_failed_api_call(self, single_position_order):
        """测试失败的API调用"""
        config = ExecutorConfig(
            java_api_base_url="http://localhost:8080",
            dry_run=False,
            max_retries=0,  # 禁用重试
        )
        executor = LiquidationExecutor(config)

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={
            'code': 500,
            'message': 'Internal Server Error',
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        with patch.object(executor, '_get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await executor.execute(single_position_order)

        assert result.failed_closes == 1
        assert result.status == ExecutionStatus.FAILED


# ============ 重试机制测试 ============

class TestRetryMechanism:
    """测试重试机制"""

    def test_is_retryable_error_500(self, executor):
        """测试500错误可重试"""
        result = {'code': 500, 'message': 'Server Error'}
        assert executor._is_retryable_error(result) is True

    def test_is_retryable_error_timeout(self, executor):
        """测试超时错误可重试"""
        result = {'code': 408, 'message': 'Request timeout'}
        assert executor._is_retryable_error(result) is True

    def test_is_retryable_error_400(self, executor):
        """测试400错误不可重试"""
        result = {'code': 400, 'message': 'Bad Request'}
        assert executor._is_retryable_error(result) is False

    def test_is_retryable_error_concurrent(self, executor):
        """测试并发错误可重试"""
        result = {'code': 409, 'message': '操作并发冲突，请重试'}
        assert executor._is_retryable_error(result) is True


# ============ 并发控制测试 ============

class TestConcurrencyControl:
    """测试并发控制"""

    @pytest.mark.asyncio
    async def test_semaphore_limit(self):
        """测试信号量限制"""
        config = ExecutorConfig(
            dry_run=True,
            max_concurrent_closes=2,
        )
        executor = LiquidationExecutor(config)

        order = MockLiquidationOrder(
            positions_to_close=[
                {'position_id': i, 'symbol': f'SYM{i}', 'quantity': 10}
                for i in range(10)
            ]
        )

        result = await executor.execute(order)

        assert result.successful_closes == 10
        assert result.status == ExecutionStatus.COMPLETED


# ============ 审计日志测试 ============

class TestAuditLog:
    """测试审计日志"""

    @pytest.mark.asyncio
    async def test_execution_started_log(self, executor, mock_order):
        """测试执行开始日志"""
        result = await executor.execute(mock_order)

        assert any(
            log['action'] == 'execution_started'
            for log in result.audit_log
        )

    @pytest.mark.asyncio
    async def test_execution_completed_log(self, executor, mock_order):
        """测试执行完成日志"""
        result = await executor.execute(mock_order)

        assert any(
            log['action'] == 'execution_completed'
            for log in result.audit_log
        )

    @pytest.mark.asyncio
    async def test_validation_passed_log(self, executor, mock_order):
        """测试验证通过日志"""
        result = await executor.execute(mock_order)

        assert any(
            log['action'] == 'validation_passed'
            for log in result.audit_log
        )


# ============ 清理测试 ============

class TestCleanup:
    """测试清理"""

    @pytest.mark.asyncio
    async def test_close_executor(self, executor):
        """测试关闭执行器"""
        # 创建一个session
        executor._session = aiohttp.ClientSession()

        await executor.close()

        assert executor._session is None or executor._session.closed


# ============ 边缘情况测试 ============

class TestEdgeCases:
    """测试边缘情况"""

    @pytest.mark.asyncio
    async def test_very_large_order(self, executor):
        """测试大量持仓订单"""
        order = MockLiquidationOrder(
            positions_to_close=[
                {'position_id': i, 'symbol': f'SYM{i}', 'quantity': 10}
                for i in range(100)
            ]
        )

        result = await executor.execute(order)

        assert result.successful_closes == 100
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_zero_quantity_position(self, executor):
        """测试零数量持仓"""
        order = MockLiquidationOrder(
            positions_to_close=[
                {
                    'position_id': 1001,
                    'symbol': 'AAPL',
                    'quantity': 0,
                    'current_price': 150.0,
                }
            ]
        )

        result = await executor.execute(order)

        assert result.successful_closes == 1

    @pytest.mark.asyncio
    async def test_negative_pnl(self, executor):
        """测试负盈亏"""
        order = MockLiquidationOrder(
            positions_to_close=[
                {
                    'position_id': 1001,
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'unrealized_pnl': -5000.0,
                }
            ]
        )

        result = await executor.execute(order)

        assert result.total_realized_pnl == -5000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
