import os
import sys
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
# MODEL_NAME 改為動態判斷

# 💡 動態路徑，自動對齊專案內 model_repository 目錄
MODEL_REPOSITORY_PATH = str(current_dir.parent / "model_repository")
LOCAL_MODELS_BASE = current_dir.parent / "active_learning_dataset" / "models"

def get_latest_model_key_from_local(model_type: str = "yolo_pose") -> str:
    """
    從地端 Fall/active_learning_dataset/models/<model_type>/ 搜尋最新重訓的最佳模型
    """
    target_dir = LOCAL_MODELS_BASE / model_type
    print(f"🔍 正在從地端 {target_dir} 搜尋最新重訓模型的權重...")
    if not target_dir.exists():
        target_dir = LOCAL_MODELS_BASE  # 相容預設路徑
        if not target_dir.exists(): return ""
    
    files = []
    for root, _, filenames in os.walk(str(target_dir)):
        for f in filenames:
            if f.endswith(".pt") or f.endswith(".onnx"):
                full_path = os.path.join(root, f)
                files.append((os.path.getmtime(full_path), full_path))
    
    if not files: return ""
def get_latest_model_key_from_local(model_type: str = "rt_detr") -> str:
    """
    從地端搜尋最新重訓的最佳模型
    """
    if model_type == "action_classifier":
        target_dir = current_dir.parent / "models" / "action_classifier"
    else:
        target_dir = LOCAL_MODELS_BASE / model_type
        
    print(f"🔍 正在從地端目錄 {target_dir} 搜尋最新重訓模型...")
    if not target_dir.exists():
        if model_type != "action_classifier":
            target_dir = LOCAL_MODELS_BASE
        if not target_dir.exists(): return ""
    
    files = []
    for root, _, filenames in os.walk(str(target_dir)):
        for f in filenames:
            if f.endswith(".pt") or f.endswith(".onnx"):
                full_path = os.path.join(root, f)
                files.append((os.path.getmtime(full_path), full_path))
    
    if not files:
        print("❌ 在地端指定目錄下找不到任何模型檔案 (.onnx 或 .pt)！")
        return ""
    
    files.sort(key=lambda x: x[0], reverse=True)
    latest_path = files[0][1]
    print(f"🎯 偵測到地端最新模型！\n   📂 本地路徑: {latest_path}")
    return latest_path


def deploy_new_model(new_model_path: str, version: int = 2, triton_model_name: str = "rt_detr"):
    """
    將地端新訓練好的模型部署至 Triton，並觸發熱部署 (Warm Start)
    """
    print(f"🚀 [MLOps 部署代理人] 開始啟動地端熱部署流程 ({triton_model_name})...")

    # 1. 確保 Triton 的目標版本資料夾存在
    target_version_dir = os.path.join(MODEL_REPOSITORY_PATH, triton_model_name, str(version))
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
    reload_url = f"{TRITON_HTTP_URL}/v2/repository/models/{triton_model_name}/load"
    
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
    status_url = f"{TRITON_HTTP_URL}/v2/models/{triton_model_name}/versions/{version}/ready"
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


def main():
    print("=======================================================")
    print("🤖 [MLOps 部署代理人] 啟動自動化地端熱部署流程")
    print("=======================================================")

    m_type = sys.argv[1] if len(sys.argv) > 1 else "rt_detr"
    
    # 決定 Triton 模型名稱
    if m_type == "action_classifier":
        triton_model_name = "action_transformer"
    elif m_type == "yolo_pose":
        triton_model_name = "yolo_pose"
    else:
        triton_model_name = "rt_detr"
        
    latest_local_path = get_latest_model_key_from_local(model_type=m_type)
    
    if latest_local_path and os.path.exists(latest_local_path):
        ext = os.path.splitext(latest_local_path)[1]
        local_temp_base = os.path.join(current_dir, "temp_downloaded_model")
        final_onnx_path = None
        
        if ext == ".pt":
            print("🔄 偵測到地端模型為 PyTorch (.pt) 格式，正在發動 ONNX 自動導出流程...")
            expected_onnx = f"{local_temp_base}.onnx"
            try:
                result = subprocess.run(
                    ["yolo", "export", f"model={latest_local_path}", "format=onnx", "imgsz=640", "opset=16"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                generated_onnx = latest_local_path.replace(".pt", ".onnx")
                if os.path.exists(generated_onnx):
                    shutil.move(generated_onnx, expected_onnx)
                    print("✅ ONNX 模型已在地端轉換完成（Opset=16 相容模式）！")
                    final_onnx_path = expected_onnx
                elif os.path.exists(expected_onnx):
                    final_onnx_path = expected_onnx
                else:
                    print(f"❌ 轉檔失敗。日誌: {result.stderr}")
            except Exception as e:
                print(f"❌ 呼叫 yolo export 時發生異常: {e}")
        else:
            final_onnx_path = latest_local_path
        
        if final_onnx_path and os.path.exists(final_onnx_path):
            deploy_new_model(new_model_path=final_onnx_path, version=2, triton_model_name=triton_model_name)
        else:
            print("❌ 部署中斷，找不到可用的 ONNX 模型檔。")
    else:
        print("ℹ️ 地端 active_learning_dataset/models/ 中尚未發現重訓產出的新模型。")

if __name__ == "__main__":
    main()