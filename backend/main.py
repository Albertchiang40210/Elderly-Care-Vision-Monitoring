import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, Depends, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


from database import Base, engine, get_db
from models import User
from security import verify_password, hash_password
from auth import create_access_token
from dependencies import get_current_user, require_admin
from event_routes import router as event_router


# ── 定義「POST /register 收到的 JSON 格式」────────────────
# 前端送來 {"username": "alice", "password": "1234"}
# FastAPI 會自動比對這個格式
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str


# 程式啟動時建立所有還不存在的資料表（表名見 models.py，例如 user_account）
# CI／測試環境設 SKIP_DB_INIT=1 時跳過：避免 import 當下就去連正式資料庫
# （雲端測試機連不到 AWS RDS，這行會卡住或報錯）
if os.getenv("SKIP_DB_INIT") != "1":
    Base.metadata.create_all(bind=engine)

# ── 建立 FastAPI app 本體 ─────────────────────────────────
# 這個 app 物件就是整個服務的核心
# 所有路由都掛在它身上（@app.post、@app.get...）
app = FastAPI()

# 告訴瀏覽器：「哪些網站可以存取我的 API。」(生產環境應該設定白名單)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域例外處理 (Global Exception Handler)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未預期的伺服器錯誤: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc) if os.getenv("DEBUG", "0") == "1" else "請聯絡系統管理員"}
    )

from fastapi.staticfiles import StaticFiles

# 挂載本地快照與影片資料夾，供前端直接讀取最新即時擷取檔案
# 生產環境建議從環境變數讀取路徑，避免 Docker 內絕對路徑失效
images_dir = os.getenv("STATIC_IMAGES_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "images")))
fall_images_dir = os.getenv("FALL_IMAGES_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Fall", "active_learning_dataset", "images")))
os.makedirs(images_dir, exist_ok=True)
os.makedirs(fall_images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")
app.mount("/fall_images", StaticFiles(directory=fall_images_dir), name="fall_images")

try:
    from backend.events.router import router as event_router
    from backend.devices.router import router as devices_router
    from backend.users.router import router as users_router
    from backend.reports.router import router as reports_router
except ModuleNotFoundError:
    from events.router import router as event_router
    from devices.router import router as devices_router
    from users.router import router as users_router
    from reports.router import router as reports_router


app.include_router(event_router)
app.include_router(devices_router)
app.include_router(users_router)
app.include_router(reports_router)






# =========================================================================
# 💡 [檔案說明與核心職責]
# 「它是後端 RESTful API 服務的核心入口 (FastAPI Main Entry point)。」
# 本檔案設定 FastAPI 應用程式實例、跨域 CORS 中間件與 API 路由掛載：
# 1. 處理前端發起的身份驗證 (JWT Token 簽發與登入)。
# 2. 提供即時告警事件查詢、護理報告拉取與設備狀態管理 API。
# 3. 整合 Server-Sent Events (SSE) 實現後端推播告警至前端網頁。
# =========================================================================

