# event_routes.py
# 事件相關的所有路由。用 APIRouter 分檔，main.py 保持乾淨
import asyncio
import os
from datetime import datetime
from typing import Literal, Optional

import boto3

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import decode_access_token
from database import get_db
from dependencies import get_current_user
from event_service import handle_incoming_event, serialize_event, watch_delivery, DeviceNotFoundError
from models import DetectEvent, Device, Staff
from sse import pool, format_sse

router = APIRouter()


# ── 機器驗證：判斷層帶 X-API-Key，跟 .env 的 EVENT_API_KEY 比對 ──
def require_api_key(x_api_key: Optional[str] = Header(None)):
    expected = os.environ.get("EVENT_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key 無效或未提供")


# ── POST /events 收到的 JSON 格式 ──
# 注意：沒有 status 欄位——status 一律由後端設 pending，不接受外部指定（spec 規定）
class EventCreateRequest(BaseModel):
    device_id: int
    event_type: str
    clip_path: Optional[str] = ""
    detected_at: datetime
    snapshot_path: Optional[str] = None
    yolo_score: Optional[float] = None
    vlm_summary: Optional[str] = None


# ════════════════════════════════════════════════════════
# POST /events（判斷層專用，API Key 驗證）
# ════════════════════════════════════════════════════════
# async def 的原因：廣播（put_nowait）要在事件迴圈執行緒上跑才安全
@router.post("/events", status_code=201, dependencies=[Depends(require_api_key)])
async def create_event(body: EventCreateRequest, db: Session = Depends(get_db)):
    try:
        # model_dump() 把 Pydantic 物件轉成 dict，交給共用處理函式
        payload = handle_incoming_event(db, body.model_dump())
    except DeviceNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # 廣播後啟動背景任務盯送達。放這裡而非 handle_incoming_event 內：
    # 路由是 async、有 event loop；handle_incoming_event 是同步、被測試直接呼叫時沒 loop
    asyncio.create_task(watch_delivery(payload["event_id"]))
    return payload


# ════════════════════════════════════════════════════════
# POST /events/{event_id}/ack（登入即可）：前端收到 SSE 後自動回報收到
# ════════════════════════════════════════════════════════
@router.post("/events/{event_id}/ack")
def ack_event(
    event_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = db.query(DetectEvent).filter(DetectEvent.event_id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")

    # 只蓋第一次：已有值就不動，保留最早的送達時間（重推可能觸發多次 ack）
    # 送達狀態記在後端 DB，給重推計時器判斷用（整套推送→ack→重推是 at-least-once 保證送達）
    # 前端打完 ack 不需要處理回應，回個小確認即可
    if event.notified_at is None:
        event.notified_at = datetime.now()
        db.commit()

    return {"status": "ok"}


# ════════════════════════════════════════════════════════
# GET /events（登入即可）：事件列表，新到舊
# ════════════════════════════════════════════════════════
@router.get("/events")
def list_events(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # JOIN 裝置表，一次查好裝置名稱/位置，跟 SSE 廣播用同一個序列化函式
    rows = (
        db.query(DetectEvent, Device)
        .join(Device, DetectEvent.device_id == Device.device_id)
        .order_by(DetectEvent.detected_at.desc())
        .all()
    )
    return [serialize_event(event, device) for event, device in rows]


# ════════════════════════════════════════════════════════
# GET /staff（登入即可）：照護員名單（指派下拉選單用）
# ════════════════════════════════════════════════════════
@router.get("/staff")
def list_staff(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return [
        {"staff_id": s.staff_id, "staff_name": s.staff_name}
        for s in db.query(Staff).order_by(Staff.staff_id).all()
    ]


# ── PATCH /events/{id}/verdict 收到的 JSON 格式 ──
class VerdictRequest(BaseModel):
    verdict: Literal["true_alarm", "false_alarm"]
    staff_id: Optional[int] = None  # 只有判真跌倒時必填


# ════════════════════════════════════════════════════════
# PATCH /events/{event_id}/verdict（登入即可）：人工判定
# ════════════════════════════════════════════════════════
@router.patch("/events/{event_id}/verdict")
async def verdict_event(
    event_id: str,
    body: VerdictRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = db.query(DetectEvent).filter(DetectEvent.event_id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")

    # 狀態轉換守門：只有 pending 能被判定（409 = 請求沒錯，但跟目前狀態衝突）
    if event.status != "pending":
        raise HTTPException(status_code=409, detail="事件已被判定過")

    if body.verdict == "true_alarm":
        # 接手 / 判定真跌倒：若前端未傳 staff_id 則預設自動填入 1 號護理師
        target_staff_id = body.staff_id if body.staff_id is not None else 1
        staff = db.query(Staff).filter(Staff.staff_id == target_staff_id).first()
        if staff is None:
            # 建立預設測試照護員
            staff = Staff(staff_id=1, staff_name="值班護理師", company_id=1)
            db.add(staff)
            db.commit()
            db.refresh(staff)
        event.status = "in_progress"
        event.verdict = "true_alarm"
        event.staff_id = staff.staff_id
    else:
        # 誤報：不用派人，直接結案（staff_id 留空）
        event.status = "resolved"
        event.verdict = "false_alarm"

    db.commit()
    db.refresh(event)

    # 先存後播：commit 成功才廣播，讓所有中控站畫面同步
    device = db.query(Device).filter(Device.device_id == event.device_id).first()
    payload = serialize_event(event, device)
    pool.broadcast("event_updated", payload)
    return payload


# ════════════════════════════════════════════════════════
# PATCH /events/{event_id}/resolve（登入即可）：結案
# ════════════════════════════════════════════════════════
@router.patch("/events/{event_id}/resolve")
async def resolve_event(
    event_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = db.query(DetectEvent).filter(DetectEvent.event_id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")

    # 只有「處理中」能結案：pending 還沒判定、resolved 已經結過了
    if event.status != "in_progress":
        raise HTTPException(status_code=409, detail="只有處理中的事件可以結案")

    event.status = "resolved"
    db.commit()
    db.refresh(event)

    device = db.query(Device).filter(Device.device_id == event.device_id).first()
    payload = serialize_event(event, device)
    pool.broadcast("event_updated", payload)
    return payload


# ════════════════════════════════════════════════════════
# GET /events/{event_id}/media（登入即可）：取得事件快照及影片 AWS S3 Presigned URL
# ════════════════════════════════════════════════════════
@router.get("/events/{event_id}/media")
def get_event_media(
    event_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = db.query(DetectEvent).filter(DetectEvent.event_id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")
    
    clip_url = None
    snapshot_url = None

    # 初始化 S3 Client (帶入 ap-northeast-1 區域)
    try:
        s3 = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1"))
    except Exception:
        s3 = None

    def _resolve_s3(uri: str) -> Optional[str]:
        if not uri or uri.strip() == "":
            return None
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri
        if uri.startswith("s3://"):
            # 💡 直接透過後端代理串流，避開瀏覽器 S3 Presigned HTTP 403 權限拒絕
            return f"/api/events/{event_id}/video"
        filename = uri.split("/")[-1]
        return f"/api/images/{filename}"

    clip_url = _resolve_s3(event.clip_path)
    snapshot_url = _resolve_s3(event.snapshot_path)

    return {
        "clip_url": clip_url,
        "snapshot_url": snapshot_url,
    }


# ════════════════════════════════════════════════════════
# GET /events/{event_id}/video：安全串流 S3 事發影片
# Safari 要求 Content-Length + Range（206）才肯播 <video>
# ════════════════════════════════════════════════════════
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

@router.get("/events/{event_id}/video")
def stream_event_video(
    event_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = db.query(DetectEvent).filter(DetectEvent.event_id == event_id).first()
    if event is None or not event.clip_path:
        raise HTTPException(status_code=404, detail="影片不存在")

    uri = event.clip_path
    video_bytes: bytes | None = None

    if uri.startswith("s3://"):
        parts = uri.replace("s3://", "").split("/", 1)
        if len(parts) == 2:
            bucket, key = parts[0], parts[1]
            try:
                s3 = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1"))
                obj = s3.get_object(Bucket=bucket, Key=key)
                video_bytes = obj['Body'].read()
            except Exception as e:
                print(f"S3 download error: {e}")
                raise HTTPException(status_code=404, detail="無法下載 S3 影片")
    else:
        filename = uri.split("/")[-1]
        local_path = os.path.join("/app/Fall/active_learning_dataset/images", filename)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                video_bytes = f.read()

    if video_bytes is None:
        raise HTTPException(status_code=404, detail="找不到影片檔案")

    total = len(video_bytes)
    range_header = request.headers.get("range")

    if range_header:
        # 解析 Range: bytes=0-  或 bytes=0-1023
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else total - 1
        end = min(end, total - 1)
        chunk = video_bytes[start:end + 1]
        return Response(
            content=chunk,
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(chunk)),
            },
        )

    return Response(
        content=video_bytes,
        status_code=200,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total),
        },
    )



# ════════════════════════════════════════════════════════
# DELETE /events（公開/測試用）：清空所有歷史與未處理事件
# ════════════════════════════════════════════════════════
@router.delete("/events")
def clear_all_events(db: Session = Depends(get_db)):
    db.query(DetectEvent).delete()
    db.commit()
    return {"message": "已清空所有事件"}


# ── SSE 專用驗證：EventSource 不能自訂 header，token 改放網址參數 ──
# 同一張 JWT，只是改插的位置；驗證邏輯用同一個 decode_access_token
def get_user_from_query_token(token: Optional[str] = Query(None)):
    if token is None:
        raise HTTPException(status_code=401, detail="缺少 token")
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="token 無效或過期")
    return payload


# ════════════════════════════════════════════════════════
# GET /stream（登入即可，token 放 query）：SSE 長連線
# ════════════════════════════════════════════════════════
@router.get("/stream")
async def stream(current_user: dict = Depends(get_user_from_query_token)):
    q = pool.register()  # 進來就掛一個信箱到連線池

    async def event_generator():
        try:
            while True:
                try:
                    # 守在自己的信箱旁等訊息，最多等 15 秒
                    message = await asyncio.wait_for(q.get(), timeout=15)
                    yield format_sse(message)
                except asyncio.TimeoutError:
                    # 15 秒沒事件 → 送心跳，防止中間網路設備掐斷「太久沒動靜」的連線
                    yield ": ping\n\n"
        finally:
            # 瀏覽器關掉/斷線/F5 → generator 被取消 → 把信箱移出連線池
            pool.unregister(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ════════════════════════════════════════════════════════
# 即時偵測廣播：POST /live-detection（推理引擎推送）
#              GET  /live-detection/stream（前端 SSE 訂閱）
# 只傳 bbox + keypoints，不落 DB，不影響其他流程
# ════════════════════════════════════════════════════════
import json as _json

_detection_subscribers: list[asyncio.Queue] = []


class LiveDetectionPayload(BaseModel):
    camera: str = "cam_in"
    persons: list  # [{bbox:[x1,y1,x2,y2], conf:float, kps:[[x,y,v]×17]}]


@router.post("/events/live-detection", status_code=204)
async def push_live_detection(payload: LiveDetectionPayload,
                              x_api_key: str = Header(..., alias="X-API-Key")):
    """推理引擎呼叫：把這幀的偵測結果廣播給所有前端訂閱者"""
    expected_key = os.environ.get("EVENT_API_KEY", "nAK4h8ARAJMjCSoWJ-uErx2KyZKGDF-jcXqmMUpkM_o")
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="無效 API Key")
    msg = f"data: {_json.dumps({'persons': payload.persons}, ensure_ascii=False)}\n\n"
    dead = []
    for q in _detection_subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _detection_subscribers.remove(q)


@router.get("/events/live-detection/stream")
async def stream_live_detection():
    """前端訂閱：SSE 即時接收每幀 bbox + keypoints"""
    q: asyncio.Queue = asyncio.Queue(maxsize=5)
    _detection_subscribers.append(q)

    async def generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=10)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            try:
                _detection_subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
