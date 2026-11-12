"""
强平执行器 - Liquidation Executor

实现真正的强平订单执行，通过调用Java后端API关闭持仓。

核心功能:
1. 接收LiquidationOrder并执行持仓平仓
2. 与Java后端PositionController集成
3. 执行状态跟踪和错误处理
4. 完整审计日志
5. 并发执行控制
6. 重试机制

使用方式:
    executor = LiquidationExecutor(java_api_base_url, auth_token)
    result = await executor.execute(liquidation_order)

作者: TraderX Team
创建: 2025-12-23
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"         # 等待执行
    VALIDATING = "validating"   # 验证中
    EXECUTING = "executing"     # 执行中
    PARTIAL = "partial"         # 部分成功
    COMPLETED = "completed"     # 全部完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消
    TIMEOUT = "timeout"         # 超时


class CloseOrderType(str, Enum):
    """平仓订单类型"""
    MARKET = "market"           # 市价平仓
    LIMIT = "limit"             # 限价平仓
    STOP = "stop"               # 止损平仓
    IMMEDIATE = "immediate"     # 立即平仓(最高优先级)


# ============ 数据类定义 ============

@dataclass
class PositionCloseResult:
    """单个持仓平仓结果"""
    position_id: int
    symbol: str
    success: bool = False

    # 执行详情
    closed_quantity: float = 0.0
    close_price: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0

    # 错误信息
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # 时间戳
    executed_at: Optional[datetime] = None

    # API响应
    api_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'success': self.success,
            'closed_quantity': self.closed_quantity,
            'close_price': self.close_price,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
        }


@dataclass
class LiquidationExecutionResult:
    """强平执行结果汇总"""
    order_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING

    # 统计
    total_positions: int = 0
    successful_closes: int = 0
    failed_closes: int = 0

    # 财务汇总
    total_realized_pnl: float = 0.0
    total_commission: float = 0.0
    margin_released: float = 0.0

    # 详细结果
    position_results: List[PositionCloseResult] = field(default_factory=list)

    # 时间戳
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 错误汇总
    errors: List[str] = field(default_factory=list)

    # 审计日志
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否全部成功"""
        return self.status == ExecutionStatus.COMPLETED and self.failed_closes == 0

    @property
    def partial_success(self) -> bool:
        """是否部分成功"""
        return self.successful_closes > 0 and self.failed_closes > 0

    def add_audit_log(self, action: str, details: Dict[str, Any]):
        """添加审计日志"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'status': self.status.value,
            'success': self.success,
            'total_positions': self.total_positions,
            'successful_closes': self.successful_closes,
            'failed_closes': self.failed_closes,
            'total_realized_pnl': self.total_realized_pnl,
            'total_commission': self.total_commission,
            'margin_released': self.margin_released,
            'position_results': [r.to_dict() for r in self.position_results],
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'errors': self.errors,
            'audit_log': self.audit_log,
        }


@dataclass
class ExecutorConfig:
    """执行器配置"""
    # API配置
    java_api_base_url: str = "http://localhost:8080"
    auth_token: Optional[str] = None

    # 超时配置
    request_timeout_seconds: float = 30.0
    total_timeout_seconds: float = 300.0  # 5分钟总超时

    # 重试配置
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0

    # 并发配置
    max_concurrent_closes: int = 5  # 最大并发平仓数

    # 安全配置
    dry_run: bool = False  # 模拟执行(不实际发送请求)
    require_confirmation: bool = False  # 是否需要确认

    # 日志配置
    log_api_requests: bool = True
    log_api_responses: bool = True


# ============ 主类: 强平执行器 ============

class LiquidationExecutor:
    """
    强平执行器

    负责执行实际的持仓平仓操作，通过调用Java后端API完成。

    核心特性:
    1. 异步并发执行
    2. 自动重试机制
    3. 完整审计日志
    4. 优雅错误处理
    5. 执行状态跟踪

    使用示例:
        config = ExecutorConfig(
            java_api_base_url="http://localhost:8080",
            auth_token="Bearer xxx"
        )
        executor = LiquidationExecutor(config)

        result = await executor.execute(liquidation_order)
        print(f"Closed {result.successful_closes}/{result.total_positions} positions")
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        on_position_closed: Optional[Callable] = None,
        on_execution_complete: Optional[Callable] = None,
    ):
        """
        初始化执行器

        Args:
            config: 执行器配置
            on_position_closed: 单个持仓平仓完成回调
            on_execution_complete: 全部执行完成回调
        """
        self.config = config or ExecutorConfig()
        self.on_position_closed = on_position_closed
        self.on_execution_complete = on_execution_complete

        # HTTP会话
        self._session: Optional[aiohttp.ClientSession] = None

        # 执行统计
        self._execution_count = 0
        self._total_positions_closed = 0
        self._total_pnl_realized = 0.0

        # 并发控制
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_closes)

    # ============ 会话管理 ============

    @asynccontextmanager
    async def _get_session(self):
        """获取HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        try:
            yield self._session
        except Exception:
            raise

    async def close(self):
        """关闭执行器"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ============ 主执行方法 ============

    async def execute(self, order) -> LiquidationExecutionResult:
        """
        执行强平订单

        Args:
            order: LiquidationOrder对象，包含要平仓的持仓列表

        Returns:
            LiquidationExecutionResult: 执行结果
        """
        result = LiquidationExecutionResult(
            order_id=order.order_id,
            started_at=datetime.now(),
        )

        result.add_audit_log("execution_started", {
            "order_id": order.order_id,
            "reason": order.reason.value if hasattr(order.reason, 'value') else str(order.reason),
            "positions_count": len(order.positions_to_close),
            "dry_run": self.config.dry_run,
        })

        logger.info(f"Starting liquidation execution: order_id={order.order_id}, "
                   f"positions={len(order.positions_to_close)}, dry_run={self.config.dry_run}")

        try:
            # 验证阶段
            result.status = ExecutionStatus.VALIDATING
            validation_ok = await self._validate_order(order, result)

            if not validation_ok:
                result.status = ExecutionStatus.FAILED
                result.completed_at = datetime.now()
                return result

            # 执行阶段
            result.status = ExecutionStatus.EXECUTING
            result.total_positions = len(order.positions_to_close)

            if self.config.dry_run:
                # 模拟执行
                await self._dry_run_execution(order, result)
            else:
                # 真实执行
                await self._real_execution(order, result)

            # 计算最终状态
            self._finalize_result(result)

        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.errors.append("Execution timed out")
            result.add_audit_log("timeout", {"timeout_seconds": self.config.total_timeout_seconds})
            logger.error(f"Liquidation execution timed out: {order.order_id}")

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.errors.append(str(e))
            result.add_audit_log("execution_error", {"error": str(e)})
            logger.exception(f"Liquidation execution failed: {order.order_id}")

        finally:
            result.completed_at = datetime.now()
            self._execution_count += 1

            result.add_audit_log("execution_completed", {
                "status": result.status.value,
                "successful": result.successful_closes,
                "failed": result.failed_closes,
                "total_pnl": result.total_realized_pnl,
            })

            # 触发完成回调
            if self.on_execution_complete:
                try:
                    await self.on_execution_complete(result)
                except Exception as e:
                    logger.error(f"Execution complete callback failed: {e}")

        return result

    # ============ 验证阶段 ============

    async def _validate_order(self, order, result: LiquidationExecutionResult) -> bool:
        """验证强平订单"""
        # 检查持仓列表
        if not order.positions_to_close:
            result.errors.append("No positions to close")
            result.add_audit_log("validation_failed", {"reason": "empty_positions"})
            return False

        # 检查每个持仓的必要字段
        for pos in order.positions_to_close:
            if 'position_id' not in pos and 'symbol' not in pos:
                result.errors.append(f"Position missing required fields: {pos}")
                result.add_audit_log("validation_failed", {
                    "reason": "missing_fields",
                    "position": pos,
                })
                return False

        result.add_audit_log("validation_passed", {
            "positions_count": len(order.positions_to_close),
        })

        return True

    # ============ 真实执行 ============

    async def _real_execution(self, order, result: LiquidationExecutionResult):
        """真实执行平仓"""
        positions = order.positions_to_close

        # 使用信号量控制并发
        tasks = []
        for pos in positions:
            task = asyncio.create_task(
                self._close_single_position_with_semaphore(pos, result)
            )
            tasks.append(task)

        # 等待所有任务完成(带总超时)
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.total_timeout_seconds
            )
        except asyncio.TimeoutError:
            # 取消未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def _close_single_position_with_semaphore(
        self,
        position: Dict[str, Any],
        result: LiquidationExecutionResult
    ):
        """使用信号量控制并发的单持仓平仓"""
        async with self._semaphore:
            await self._close_single_position(position, result)

    async def _close_single_position(
        self,
        position: Dict[str, Any],
        result: LiquidationExecutionResult
    ):
        """执行单个持仓的平仓"""
        position_id = position.get('position_id')
        symbol = position.get('symbol', 'UNKNOWN')

        pos_result = PositionCloseResult(
            position_id=position_id,
            symbol=symbol,
        )

        logger.info(f"Closing position: id={position_id}, symbol={symbol}")

        retries = 0
        delay = self.config.retry_delay_seconds

        while retries <= self.config.max_retries:
            try:
                # 调用Java后端API
                api_result = await self._call_close_api(position)

                if api_result.get('success', False):
                    # 成功
                    pos_result.success = True
                    pos_result.closed_quantity = float(position.get('quantity', 0))
                    pos_result.close_price = float(api_result.get('close_price', 0))
                    pos_result.realized_pnl = float(api_result.get('realized_pnl', 0))
                    pos_result.commission = float(api_result.get('commission', 0))
                    pos_result.executed_at = datetime.now()
                    pos_result.api_response = api_result

                    result.successful_closes += 1
                    result.total_realized_pnl += pos_result.realized_pnl
                    result.total_commission += pos_result.commission
                    result.margin_released += float(position.get('margin_to_release', 0))

                    self._total_positions_closed += 1
                    self._total_pnl_realized += pos_result.realized_pnl

                    logger.info(f"Position closed successfully: {symbol}, "
                               f"pnl={pos_result.realized_pnl:.2f}")
                    break

                else:
                    # API返回失败
                    error_msg = api_result.get('message', 'Unknown error')
                    pos_result.error_code = str(api_result.get('code', 'UNKNOWN'))
                    pos_result.error_message = error_msg

                    # 检查是否可重试
                    if self._is_retryable_error(api_result):
                        retries += 1
                        if retries <= self.config.max_retries:
                            logger.warning(f"Retrying position close: {symbol}, "
                                         f"attempt {retries}/{self.config.max_retries}")
                            await asyncio.sleep(delay)
                            delay *= self.config.retry_backoff_factor
                            continue

                    # 不可重试或已耗尽重试次数
                    result.failed_closes += 1
                    result.errors.append(f"Failed to close {symbol}: {error_msg}")
                    logger.error(f"Position close failed: {symbol}, error={error_msg}")
                    break

            except aiohttp.ClientError as e:
                retries += 1
                if retries <= self.config.max_retries:
                    logger.warning(f"Network error closing {symbol}, "
                                 f"retrying ({retries}/{self.config.max_retries}): {e}")
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff_factor
                else:
                    pos_result.error_code = "NETWORK_ERROR"
                    pos_result.error_message = str(e)
                    result.failed_closes += 1
                    result.errors.append(f"Network error closing {symbol}: {e}")
                    logger.error(f"Position close network error: {symbol}, error={e}")

            except Exception as e:
                pos_result.error_code = "INTERNAL_ERROR"
                pos_result.error_message = str(e)
                result.failed_closes += 1
                result.errors.append(f"Internal error closing {symbol}: {e}")
                logger.exception(f"Position close internal error: {symbol}")
                break

        result.position_results.append(pos_result)

        # 触发单持仓完成回调
        if self.on_position_closed:
            try:
                await self.on_position_closed(pos_result)
            except Exception as e:
                logger.error(f"Position closed callback failed: {e}")

    async def _call_close_api(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """调用Java后端平仓API"""
        position_id = position.get('position_id')
        url = f"{self.config.java_api_base_url}/positions/{position_id}/close"

        headers = {
            'Content-Type': 'application/json',
        }

        if self.config.auth_token:
            headers['Authorization'] = self.config.auth_token

        # 构建请求体
        payload = {
            'price': position.get('current_price'),
            'commission': 0,
            'notes': f"Liquidation - margin call",
        }

        if self.config.log_api_requests:
            logger.debug(f"API Request: POST {url}, payload={payload}")

        async with self._get_session() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()

                if self.config.log_api_responses:
                    logger.debug(f"API Response: status={response.status}, body={response_data}")

                # 解析响应
                if response.status == 200:
                    return {
                        'success': True,
                        'code': response_data.get('code', 200),
                        'message': response_data.get('message', 'Success'),
                        'close_price': position.get('current_price', 0),
                        'realized_pnl': 0,  # 从响应中解析
                        'commission': 0,
                    }
                else:
                    return {
                        'success': False,
                        'code': response.status,
                        'message': response_data.get('message', f'HTTP {response.status}'),
                    }

    def _is_retryable_error(self, api_result: Dict[str, Any]) -> bool:
        """判断是否可重试的错误"""
        code = api_result.get('code', 0)
        message = api_result.get('message', '').lower()

        # HTTP 5xx 错误可重试
        if isinstance(code, int) and 500 <= code < 600:
            return True

        # 特定错误消息可重试
        retryable_messages = [
            'timeout',
            'connection',
            '并发',
            'concurrent',
            'busy',
            'try again',
            '重试',
        ]

        for msg in retryable_messages:
            if msg in message:
                return True

        return False

    # ============ 模拟执行 ============

    async def _dry_run_execution(self, order, result: LiquidationExecutionResult):
        """模拟执行(不实际发送请求)"""
        logger.info("Dry run mode - simulating execution")

        for pos in order.positions_to_close:
            pos_result = PositionCloseResult(
                position_id=pos.get('position_id'),
                symbol=pos.get('symbol', 'UNKNOWN'),
                success=True,
                closed_quantity=float(pos.get('quantity', 0)),
                close_price=float(pos.get('current_price', 0)),
                realized_pnl=float(pos.get('unrealized_pnl', 0)),
                executed_at=datetime.now(),
            )

            result.position_results.append(pos_result)
            result.successful_closes += 1
            result.total_realized_pnl += pos_result.realized_pnl
            result.margin_released += float(pos.get('margin_to_release', 0))

            # 更新执行器统计
            self._total_positions_closed += 1
            self._total_pnl_realized += pos_result.realized_pnl

            # 触发单持仓完成回调
            if self.on_position_closed:
                try:
                    await self.on_position_closed(pos_result)
                except Exception as e:
                    logger.error(f"Position closed callback failed: {e}")

            # 模拟一点延迟
            await asyncio.sleep(0.1)

        result.add_audit_log("dry_run_completed", {
            "simulated_closes": len(order.positions_to_close),
        })

    # ============ 结果处理 ============

    def _finalize_result(self, result: LiquidationExecutionResult):
        """计算最终执行状态"""
        if result.total_positions == 0:
            # 没有持仓需要平仓 - 视为失败(应该在验证阶段拦截)
            result.status = ExecutionStatus.FAILED
        elif result.failed_closes == 0 and result.successful_closes > 0:
            result.status = ExecutionStatus.COMPLETED
        elif result.successful_closes > 0 and result.failed_closes > 0:
            result.status = ExecutionStatus.PARTIAL
        elif result.successful_closes == 0 and result.failed_closes > 0:
            result.status = ExecutionStatus.FAILED
        else:
            result.status = ExecutionStatus.COMPLETED

    # ============ 统计信息 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        return {
            'execution_count': self._execution_count,
            'total_positions_closed': self._total_positions_closed,
            'total_pnl_realized': self._total_pnl_realized,
            'config': {
                'java_api_base_url': self.config.java_api_base_url,
                'dry_run': self.config.dry_run,
                'max_concurrent': self.config.max_concurrent_closes,
                'max_retries': self.config.max_retries,
            }
        }


# ============ 工厂函数 ============

def create_liquidation_executor(
    java_api_url: str = "http://localhost:8080",
    auth_token: Optional[str] = None,
    dry_run: bool = False,
) -> LiquidationExecutor:
    """
    创建强平执行器的便捷工厂函数

    Args:
        java_api_url: Java后端API地址
        auth_token: 认证令牌
        dry_run: 是否模拟执行

    Returns:
        配置好的LiquidationExecutor实例
    """
    config = ExecutorConfig(
        java_api_base_url=java_api_url,
        auth_token=auth_token,
        dry_run=dry_run,
    )

    return LiquidationExecutor(config)


# ============ 与MarginMonitor集成 ============

def create_liquidation_callback(executor: LiquidationExecutor) -> Callable:
    """
    创建可传递给RealtimeMarginMonitor的强平回调函数

    使用示例:
        executor = create_liquidation_executor(...)
        callback = create_liquidation_callback(executor)

        monitor = RealtimeMarginMonitor(
            liquidation_callback=callback
        )
    """
    async def liquidation_callback(order) -> Dict[str, Any]:
        """强平回调函数"""
        result = await executor.execute(order)

        return {
            'success': result.success,
            'order_id': result.order_id,
            'status': result.status.value,
            'successful_closes': result.successful_closes,
            'failed_closes': result.failed_closes,
            'total_realized_pnl': result.total_realized_pnl,
            'margin_released': result.margin_released,
            'errors': result.errors,
        }

    return liquidation_callback
