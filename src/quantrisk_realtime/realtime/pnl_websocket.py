"""
实时P&L WebSocket API端点

提供WebSocket接口用于实时P&L数据推送。

端点:
- /ws/pnl/portfolio/{portfolio_id} - 订阅投资组合P&L
- /ws/pnl/position/{position_id} - 订阅单持仓P&L
- /ws/pnl/account/{account_id} - 订阅账户级别P&L

作者: TraderX Team
创建: 2025-12-24
"""

import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/ws/pnl", tags=["PnL WebSocket"])

# 全局P&L推送器实例(在应用启动时初始化)
_pnl_streamer = None


def get_pnl_streamer():
    """获取P&L推送器实例"""
    global _pnl_streamer
    if _pnl_streamer is None:
        from .pnl_streamer import create_pnl_streamer
        _pnl_streamer = create_pnl_streamer()
    return _pnl_streamer


def set_pnl_streamer(streamer):
    """设置P&L推送器实例"""
    global _pnl_streamer
    _pnl_streamer = streamer


# ============ WebSocket端点 ============

@router.websocket("/portfolio/{portfolio_id}")
async def websocket_portfolio_pnl(
    websocket: WebSocket,
    portfolio_id: int,
):
    """
    订阅投资组合P&L实时更新

    连接后将自动接收:
    1. 初始全量P&L数据
    2. 增量P&L更新(每秒或有变化时)
    3. 定时心跳(30秒)

    消息格式:
    ```json
    {
        "type": "pnl_update",
        "update_type": "incremental",
        "subscription_type": "portfolio",
        "timestamp": "2025-12-24T10:00:00",
        "sequence": 123,
        "portfolio": {
            "portfolio_id": 1,
            "total_unrealized_pnl": 1500.50,
            "total_pnl_today": 2300.00,
            ...
        },
        "pnl_change": 50.25
    }
    ```
    """
    streamer = get_pnl_streamer()

    await websocket.accept()
    logger.info(f"WebSocket connected for portfolio {portfolio_id}")

    try:
        # 订阅
        success = await streamer.subscribe_portfolio(websocket, portfolio_id)
        if not success:
            await websocket.send_json({
                "type": "error",
                "message": "Subscription failed - limit exceeded",
            })
            await websocket.close()
            return

        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "subscription_type": "portfolio",
            "portfolio_id": portfolio_id,
            "timestamp": datetime.now().isoformat(),
        })

        # 保持连接,等待断开
        while True:
            try:
                # 接收客户端消息(如ping)
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get('type') == 'ping':
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    })
                elif message.get('type') == 'unsubscribe':
                    break

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass  # 忽略无效JSON
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for portfolio {portfolio_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await streamer.handle_disconnect(websocket)


@router.websocket("/position/{position_id}")
async def websocket_position_pnl(
    websocket: WebSocket,
    position_id: int,
):
    """
    订阅单持仓P&L实时更新

    适用于需要关注单个持仓P&L变化的场景。
    """
    streamer = get_pnl_streamer()

    await websocket.accept()
    logger.info(f"WebSocket connected for position {position_id}")

    try:
        success = await streamer.subscribe_position(websocket, position_id)
        if not success:
            await websocket.send_json({
                "type": "error",
                "message": "Subscription failed",
            })
            await websocket.close()
            return

        await websocket.send_json({
            "type": "connected",
            "subscription_type": "position",
            "position_id": position_id,
            "timestamp": datetime.now().isoformat(),
        })

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get('type') == 'ping':
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    })
                elif message.get('type') == 'unsubscribe':
                    break

            except WebSocketDisconnect:
                break
            except Exception:
                logger.exception("P&L websocket message loop failed; closing connection")
                break

    except WebSocketDisconnect:
        pass
    finally:
        await streamer.handle_disconnect(websocket)


@router.websocket("/account/{account_id}")
async def websocket_account_pnl(
    websocket: WebSocket,
    account_id: str,
):
    """
    订阅账户级别P&L实时更新

    接收账户下所有投资组合的P&L汇总和变化。
    """
    streamer = get_pnl_streamer()

    await websocket.accept()
    logger.info(f"WebSocket connected for account {account_id}")

    try:
        success = await streamer.subscribe_account(websocket, account_id)
        if not success:
            await websocket.send_json({
                "type": "error",
                "message": "Subscription failed",
            })
            await websocket.close()
            return

        await websocket.send_json({
            "type": "connected",
            "subscription_type": "account",
            "account_id": account_id,
            "timestamp": datetime.now().isoformat(),
        })

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get('type') == 'ping':
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    })
                elif message.get('type') == 'unsubscribe':
                    break

            except WebSocketDisconnect:
                break
            except Exception:
                logger.exception("P&L websocket message loop failed; closing connection")
                break

    except WebSocketDisconnect:
        pass
    finally:
        await streamer.handle_disconnect(websocket)


# ============ HTTP端点(状态查询) ============

@router.get("/status")
async def get_pnl_streamer_status():
    """
    获取P&L推送服务状态

    Returns:
        推送器统计信息
    """
    streamer = get_pnl_streamer()
    return {
        "success": True,
        "data": streamer.get_statistics(),
    }


@router.post("/start")
async def start_pnl_streamer():
    """启动P&L推送服务"""
    streamer = get_pnl_streamer()

    if streamer.is_running:
        return {"success": True, "message": "Already running"}

    await streamer.start()
    return {"success": True, "message": "Started"}


@router.post("/stop")
async def stop_pnl_streamer():
    """停止P&L推送服务"""
    streamer = get_pnl_streamer()

    if not streamer.is_running:
        return {"success": True, "message": "Already stopped"}

    await streamer.stop()
    return {"success": True, "message": "Stopped"}
