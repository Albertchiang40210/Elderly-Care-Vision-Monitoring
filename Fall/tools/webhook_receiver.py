import os
import sys
import traceback
import asyncio
from typing import Dict
from fastapi import FastAPI, Request
import uvicorn

# 確保路徑正確以載入 submit_task
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    # 🎯 🌟 確保 submit_task.py 的 main 函式支援傳入 project_name 參數
    # 例如: def main(project_name="Fall_Detection"):
    from submit_task import main as trigger_clearml_training
except ImportError as e:
    print(f"❌ 無法載入 submit_task 零件: {e}")
    trigger_clearml_training = None

app = FastAPI()

# 🚀 改為「字典型態」的計數器，為每個專案獨立計數！
# 格式: {"Hazard_Detection": 3, "Fall_Detection": 8}
PROJECT_COUNTERS: Dict[str, int] = {}
TRIGGER_THRESHOLD = 100  # 🚀 累積滿 100 張照片即自動點火 ClearML 重訓

lock = asyncio.Lock()

async def async_clearml_fire(project_name: str):
    """
    使用 asyncio 異步執行，動態帶入 project_name，
    將 Task 推入 ClearML 對應專案的 Queue。
    """
    if trigger_clearml_training is None:
        print("❌ 錯誤：重訓點火零件 (submit_task.py) 未正確載入。")
        return
        
    try:
        print(f"[*] 🚀 [點火控制閥] 專案 '{project_name}' 正在為彈射任務加固環境配置...")
        
        def secured_trigger():
            import subprocess
            # 🎯 啟動重訓前，強制執行 SDK 同步，帶入對應專案
            try:
                print(f"[*] 正在連線 Label Studio 將 '{project_name}' 最新人工標記成果同步至地端...")
                sdk_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_to_labelstudio_sdk.py")
                # 可選：若你的 SDK 支援傳帶專案名稱參數，可寫成: [sys.executable, sdk_script, "--project", project_name]
                subprocess.run([sys.executable, sdk_script], check=True)
                print(f"✅ [同步成功] '{project_name}' 人工修正之 YOLO 標籤檔案已順利落地！")
            except Exception as sync_err:
                print(f"⚠️ 同步人工標記失敗 (將以地端既有快取進行重訓): {sync_err}")

            # 🎯 核心修復：將專案名稱動態傳入 ClearML Task 觸發器
            try:
                # 嘗試帶入 project_name
                trigger_clearml_training(project_name=project_name)
            except TypeError:
                # 備用機制：若你的 submit_task.py main() 還沒支援參數，退回無參數呼叫
                print("⚠️ 提示: submit_task.main() 未接受 project_name 參數，執行預設點火...")
                trigger_clearml_training()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, secured_trigger)
        
        print(f"✅ [排隊成功] 專案 '{project_name}' 任務已成功推入 ClearML 佇列！")
    except Exception as e:
        print(f"❌ [點火失敗] '{project_name}' 點火程序崩潰！詳細內容:")
        traceback.print_exc()


@app.post("/webhook")
async def label_studio_webhook(request: Request):
    global PROJECT_COUNTERS
    
    try:
        data = await request.json()
        action = data.get("action", "")
        
        # 🎯 🌟 核心提取：從 Label Studio 的 Webhook Payload 中精準解析 Project Title
        # Label Studio Payload 結構範例: {"project": {"id": 1, "title": "Hazard_Detection"}, ...}
        project_info = data.get("project", {})
        project_name = project_info.get("title", "Fall_Detection") # 預設 Fall_Detection 避險
        
    except Exception as e:
        print(f"❌ 解析 Webhook JSON 失敗: {e}")
        return {"status": "bad_request"}
    
    # 🚀 半自動主動學習閉環：監聽人工審核提交 (單張/整批 annotation_created, annotations_created, tasks_updated)
    if action and action.lower() in ["annotation_created", "annotation_updated", "annotations_created", "tasks_updated"]:
        async with lock:
            # 針對該專案初始化與累加計數
            current_count = PROJECT_COUNTERS.get(project_name, 0) + 1
            PROJECT_COUNTERS[project_name] = current_count
            
            print(f"📥 [Webhook 捕獲] 專案: '{project_name}' | 事件: {action} | 當前緩衝池: {current_count} / {TRIGGER_THRESHOLD}")
            
            # 當該專案的數量到達門檻時
            if current_count >= TRIGGER_THRESHOLD:
                print(f"🔥 [門檻達成] 專案 '{project_name}' 標註門檻達標 ({TRIGGER_THRESHOLD}張)！立即發送 ClearML 排隊重訓命令！")
                
                # 🎯 異步點火，傳入獨立的 project_name
                asyncio.create_task(async_clearml_fire(project_name))
                
                # 歸零該專案的計數器
                PROJECT_COUNTERS[project_name] = 0
    else:
        # 忽略其他無關事件（如 task_created, project_updated 等）
        pass
        
    return {"status": "processed"}


if __name__ == "__main__":
    print(f"[*] MLOps 多專案全自動點火閥已就位，監聽 Port 9001...")
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