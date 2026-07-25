import os
import shutil
import time
import requests
import subprocess
import boto3
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError

# ==================== 🎯 方案二：動態載入同目錄下的 .env ====================
# 取得目前此 Python 檔案 (model_deployment_agent.py) 的絕對路徑目錄
current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'

# 載入指定路徑的 .env 檔案（自動注入環境變數，讓 boto3 能順利讀取憑證）
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ 已成功載入專屬環境變數檔: {env_path}")
else:
    print(f"⚠️ 警告：在 {env_path} 找不到 .env 檔案，將嘗試讀取系統預設環境變數。")
# ==============================================================================

# ==================== 配置參數 ====================
TRITON_HTTP_URL = "http://localhost:8000"  # Triton HTTP 服務埠
MODEL_NAME = "rt_detr"                     # Triton 中的模型名稱

# 💡 動態路徑，自動對齊專案內 model_repository 目錄
MODEL_REPOSITORY_PATH = str(current_dir.parent / "model_repository")

# ☁️ AWS S3 實體配置（改為優先從環境變數讀取，若無則套用預設值）
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "aipe03-3")  # 你的 S3 儲存桶名稱

# 🛠️ 修正後的路徑前綴：精準對齊 ClearML 自動生成的 S3 深層結構
S3_PREFIX = "clearml-artifacts/models/fall_detection/Fall_Detection/"  
# =================================================

def get_latest_model_key_from_s3(bucket_name: str, prefix: str) -> str:
    """
    自動掃描 S3，找出 Fall_Detection/ 底下最新修改的資料夾中的 model.onnx (或權重檔)
    """
    print("🔍 正在從 AWS S3 搜尋最新的重訓模型資料夾...")
    
    # 這裡 boto3 會自動去吃剛才 load_dotenv 載入的 AWS_ACCESS_KEY_ID 等環境變數
    s3 = boto3.client('s3')
    
    # 列出該前綴下的所有物件
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    onnx_files = []
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # 尋找 ONNX 模型檔
                if key.endswith('.onnx'):
                    onnx_files.append({
                        'Key': key,
                        'LastModified': obj['LastModified']
                    })
    
    if not onnx_files:
        # 如果沒找到 .onnx，退而求其次找 .pt 檔
        print("⚠️ 沒找到 .onnx 檔，嘗試搜尋 .pt 檔...")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.pt'):
                        onnx_files.append({
                            'Key': key,
                            'LastModified': obj['LastModified']
                        })

    if not onnx_files:
        print("❌ 在 S3 指定目錄下找不到任何模型檔案 (.onnx 或 .pt)！")
        return None
    
    # 依最後修改時間排序，拿出最新上傳的模型
    onnx_files.sort(key=lambda x: x['LastModified'], reverse=True)
    latest_model = onnx_files[0]
    
    print("🎯 偵測到最新模型！")
    print(f"   📂 S3 路徑: s3://{bucket_name}/{latest_model['Key']}")
    print(f"   🕒 修改時間: {latest_model['LastModified']}")
    
    return latest_model['Key']


def download_model_from_s3(bucket_name: str, s3_key: str, local_path: str) -> bool:
    """
    從 AWS S3 下載指定模型檔案到本地
    """
    try:
        s3 = boto3.client('s3')
        print(f"📥 開始下載最新模型至本地暫存: {local_path}...")
        
        # 確保本地暫存目錄存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        s3.download_file(bucket_name, s3_key, local_path)
        print("✅ 模型下載完成！")
        return True
    except NoCredentialsError:
        print("❌ 錯誤：找不到 AWS 憑證，請確認專案根目錄中的 .env 是否正確配置。")
        return False
    except Exception as e:
        print(f"❌ 從 S3 下載模型時發生錯誤: {e}")
        return False


def deploy_new_model(new_model_path: str, version: int = 2):
    """
    將新訓練好的模型部署至 Triton，並觸發熱部署 (Warm Start)
    """
    print("🚀 [MLOps 部署代理人] 開始啟動熱部署流程...")

    # 1. 確保 Triton 的目標版本資料夾存在
    target_version_dir = os.path.join(MODEL_REPOSITORY_PATH, MODEL_NAME, str(version))
    os.makedirs(target_version_dir, exist_ok=True)
    
    target_model_path = os.path.join(target_version_dir, "model.onnx")

    # 2. 複製新模型至 Triton 目錄
    try:
        print(f"📦 正在將新模型從 {new_model_path} 複製到 {target_model_path}...")
        shutil.copy(new_model_path, target_model_path)
        print("✅ 模型檔案複製成功！")
    except Exception as e:
        print(f"❌ 檔案複製失敗: {e}")
        return False

    # 3. 發送 HTTP POST 請求給 Triton 觸發熱部署 (Warm Start)
    reload_url = f"{TRITON_HTTP_URL}/v2/repository/models/{MODEL_NAME}/load"
    
    print(f"🔄 正在向 Triton 發送重新載入訊號 (HTTP POST): {reload_url}")
    try:
        response = requests.post(reload_url)
        if response.status_code == 200:
            print(f"⚡ Triton 已接收到載入指令，開始在背景載入模型版本 {version}...")
        elif response.status_code == 400 and "polling is enabled" in response.text:
            print("ℹ️  Triton 偵測到已啟動自動輪詢 (Polling Mode)，將由 Triton 自動偵測並載入新 model。")
        else:
            print(f"❌ Triton 拒絕載入請求。狀態碼: {response.status_code}, 內容: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        # 🎯 關鍵修改：地端 Mac 無 N卡，無法與 Triton 通訊時，不拋出 Exception 崩潰，而是優雅降級
        print("ℹ️  提示：地端 Mac 無 N卡，無法連線至 Triton 伺服器，將跳過 Triton 熱部署，採用邊緣端本地 CPU 推理。")
        return True

    # 4. 主動輪詢 (Polling) 檢查模型是否成功 READY
    status_url = f"{TRITON_HTTP_URL}/v2/models/{MODEL_NAME}/versions/{version}/ready"
    max_retries = 25  # ONNX 初始化與編譯較耗時，設定為 25 次 (50秒)
    retry_interval = 2

    print("🔎 開始追蹤模型載入狀態...")
    for i in range(max_retries):
        try:
            status_response = requests.get(status_url)
            if status_response.status_code == 200:
                print(f"\n🎉 [部署成功] 新模型 (版本 {version}) 已成功上線，狀態：🟢 READY")
                print("💡 智慧病房監控已無痛切換至新大腦，服務未中斷！")
                return True
        except Exception:
            pass
        
        print(f"⏳ 載入中... ({i+1}/{max_retries})")
        time.sleep(retry_interval)

    print("\n❌ [失敗] 超時！Triton 未能在預期時間內將模型載入為 READY 狀態。請確認 Triton 的 config.pbtxt 是否正確。")
    return False


if __name__ == "__main__":
    # 暫存下載模型的基礎路徑
    local_temp_base = "./temp_downloaded_model"
    
    # 🚀 第一步：動態尋找 S3 上最新產生的模型路徑
    latest_s3_key = get_latest_model_key_from_s3(
        bucket_name=AWS_BUCKET_NAME,
        prefix=S3_PREFIX
    )
    
    # 🚀 第二步：下載最新模型並判斷是否需要轉檔
    if latest_s3_key:
        ext = os.path.splitext(latest_s3_key)[1]
        download_path = f"{local_temp_base}{ext}"
        
        success = download_model_from_s3(
            bucket_name=AWS_BUCKET_NAME,
            s3_key=latest_s3_key,
            local_path=download_path
        )
        
        if success and os.path.exists(download_path):
            final_onnx_path = None
            
            # 💡 核心轉換邏輯：如果是 .pt 檔，自動呼叫 YOLO CLI 轉成 .onnx
            if ext == ".pt":
                print("🔄 偵測到下載的模型為 PyTorch (.pt) 格式，正在呼叫轉檔指令...")
                expected_onnx = f"{local_temp_base}.onnx"
                
                try:
                    # 💡 關鍵修改：在轉檔指令最後加上 "opset=16"，強制導出相容於當前 Triton (ONNX Runtime) 的格式
                    result = subprocess.run(
                        ["yolo", "export", f"model={download_path}", "format=onnx", "imgsz=640", "opset=16"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    
                    # 檢查原位置導出成功後的檔案路徑
                    generated_onnx = download_path.replace(".pt", ".onnx")
                    
                    if os.path.exists(generated_onnx):
                        # 將轉好的模型重新命名到我們的 ONNX 暫存檔
                        shutil.move(generated_onnx, expected_onnx)
                        print("✅ ONNX 模型已成功在本地轉換完成（Opset=16 相容模式）！")
                        final_onnx_path = expected_onnx
                    else:
                        print(f"❌ 轉檔失敗，未在預期位置找到 ONNX 檔。日誌: {result.stderr}")
                except Exception as e:
                    print(f"❌ 呼叫 yolo export 時發生異常: {e}")
            else:
                # 如果從 S3 下載下來原本就是 .onnx，直接使用
                final_onnx_path = download_path
            
            # 🚀 第三步：執行熱部署
            if final_onnx_path and os.path.exists(final_onnx_path):
                deploy_new_model(new_model_path=final_onnx_path, version=2)
            else:
                print("❌ 部署中斷，找不到可用的 ONNX 模型檔。")
            
            # 🧹 清理暫存的所有本地暫存檔 (.pt 與 .onnx)
            for file in [f"{local_temp_base}.pt", f"{local_temp_base}.onnx"]:
                if os.path.exists(file):
                    os.remove(file)
            print("🧹 已清除本地所有暫存模型檔案。")
    else:
        print("❌ 無法取得最新模型路徑，部署中斷。")