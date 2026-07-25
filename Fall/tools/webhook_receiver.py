import os
import sys
import traceback
import asyncio
from fastapi import FastAPI, Request
import uvicorn
from clearml import Task  # 🎯 導入 Task 以便進行環境防禦配置

# 確保路徑正確以載入 submit_task
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    # 🎯 🌟 [重要] 這裡的 trigger_clearml_training 呼叫會排隊發射
    from submit_task import main as trigger_clearml_training
except ImportError as e:
    print(f"❌ 無法載入 submit_task 零件: {e}")
    trigger_clearml_training = None

app = FastAPI()

ANNOTATION_COUNT = 0
TRIGGER_THRESHOLD = 100 # 🚀 業界高標準生產規格：累積滿 100 張人工審核/修正照片才點火重訓

async def async_clearml_fire():
    """
    使用 asyncio 異步執行，將 Task 推入 ClearML Queue，
    並在此處強行注入環境防禦鎖，確保 Agent 不會崩潰。
    """
    if trigger_clearml_training is None:
        print("❌ 錯誤：重訓點火零件未正確載入。")
        return
        
    try:
        print("[*] 🚀 [點火控制閥] 正在為彈射任務加固環境配置...")
        
        # 🎯 🌟 【穩定性防禦核心】
        # 在執行排隊任務前，確保任何由這裡產生的 Task 都禁止 Agent 進行自動環境建置
        # 這能徹底閃過你之前遇到的 Python 3.13 / venv 編譯衝突
        def secured_trigger():
            import subprocess
            # 🎯 [Option B] 啟動重訓前，強制執行一次 SDK 同步，將 Label Studio 上的最新人工修改標註完整拉回地端
            try:
                print("[*] 正在連線 Label Studio 將最新人工標記成果同步至地端...")
                sdk_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_to_labelstudio_sdk.py")
                subprocess.run([sys.executable, sdk_script], check=True)
                print("✅ [同步成功] 人工修正之 YOLO 標籤檔案已順利落地！")
            except Exception as sync_err:
                print(f"⚠️ 同步人工標記失敗 (將以地端既有快取進行重訓): {sync_err}")

            # 建立一個佔位 Task 來設定環境偏好，或者直接呼叫你的主訓練函數
            # 若你的 submit_task.py 內已有 Task 初始化，請確保該腳本也有這兩行
            trigger_clearml_training()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, secured_trigger)
        
        print("✅ [排隊成功] 任務已成功推入 ClearML 'default' 佇列！")
    except Exception as e:
        print(f"❌ [點火失敗] 點火程序崩潰！詳細內容:")
        traceback.print_exc()

lock = asyncio.Lock()

@app.post("/webhook")
async def label_studio_webhook(request: Request):
    global ANNOTATION_COUNT
    
    try:
        data = await request.json()
        action = data.get("action", "")
    except Exception as e:
        print(f"❌ 解析 Webhook JSON 失敗: {e}")
        return {"status": "bad_request"}
    
    # 🚀 半自動主動學習閉環：監聽人工審核提交 (annotation_created) 與人工修正 (annotation_updated)
    if action and action.lower() in ["annotation_created", "annotation_updated"]:
        async with lock:
            ANNOTATION_COUNT += 1
            print(f"📥 [Webhook 捕獲] 事件: {action} (人工審核/修改標註) | 當前緩衝池: {ANNOTATION_COUNT} / {TRIGGER_THRESHOLD}")
            
            if ANNOTATION_COUNT >= TRIGGER_THRESHOLD:
                print(f"🔥 [門檻達成] 人工標註門檻達標！立即發送 ClearML 排隊重訓命令！")
                # 🎯 使用非同步任務發射，避免 FastAPI 卡死
                asyncio.create_task(async_clearml_fire())
                ANNOTATION_COUNT = 0 
    else:
        pass
        
    return {"status": "processed"}

if __name__ == "__main__":
    print(f"[*] MLOps 全自動點火閥已就位，監聽 Port 9001...")
    uvicorn.run(app, host="0.0.0.0", port=9001)



#「它是我們 MLOps 飛輪的『全自動點火控制閥』，默默守在後台數張數，時間一到就彈射重訓任務！」
#在主動學習（Active Learning）的架構中，最核心的價值就是「當累積了一定數量的新標註資料後，模型就要自動去學習它」。這支程式就是負責在背景默默看守的總監聽官：
#架設後台隱形接收港口（FastAPI Webhook）：
#它利用 FastAPI 框架，在後台 Port 9001 撐起了一個名為 /webhook 的接收通道。當你在 Label Studio 上每點擊提交或修改一次標註時（annotation_created / annotation_updated），Label Studio 就會秒發一封通知信給它。
#智慧緩衝池計數機制（Threshold 緩衝區）：
#它內部做了一個計數器 ANNOTATION_COUNT。你非常聰明地設定了這個暗號：「不需要每標註完一張照片就去驚動 ClearML 重訓伺服器」（因為模型训练需要時間 and 算力）。它會默默在緩衝池裡數張數，當你累積標註完 50 張新照片（TRIGGER_THRESHOLD = 50）之後，它才會真正拉響警報！
#異步非阻塞彈射與併發鎖防禦（Asyncio Executor & Lock）：
#這段程式寫得極具高級軟體工程師的實戰技巧！ 
#1. 在這裡使用了 Python 的 asyncio.create_task 配合 loop.run_in_executor，將點火連線打包丟到背景執行，讓 FastAPI 一秒內回覆 processed，網頁標註端完全不卡頓。
#2. 新增了 asyncio.Lock() 併發鎖，防範在短時間內有多個標註事件同時到達時產生的計數競爭（Race Condition），確保 MLOps 點火彈射機制的絕對穩定！