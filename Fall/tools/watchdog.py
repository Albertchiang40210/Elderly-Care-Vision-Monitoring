import os
import time
import subprocess
import logging
import signal
import sys

# 動態計算專案根目錄路徑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_DIR = os.path.dirname(CURRENT_DIR)

LOG_FILE = os.path.join(FALL_DIR, "watchdog.log")
INFERENCE_SCRIPT = os.path.join(CURRENT_DIR, "inference_to_labelstudio_sdk.py")
INTERVAL = 300  # 掃描間隔 5 分鐘

# 設定詳細日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger("AdvancedWatchdog")

def graceful_exit(signum, frame):
    logger.info("收到終止訊號，正在安全關閉監控服務...")
    sys.exit(0)

# 註冊訊號處理，確保系統關閉時不會產生殘留錯誤
signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

def run_sync():
    """執行同步標註腳本並處理異常"""
    logger.info(">>> 開始執行影像同步與自動標註流程 (包含 DETR 與 YOLO-Pose)...")
    try:
        # 1. 執行 DETR 物件偵測標註
        logger.info(">>> 1/2 啟動 DETR 物件偵測自動標註...")
        result_detr = subprocess.run(
            ["python", INFERENCE_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        if result_detr.stdout:
            logger.info(f"DETR 執行結果: {result_detr.stdout.strip()}")
            
        # 2. 執行 YOLO-Pose 骨架標註
        POSE_SCRIPT = os.path.join(CURRENT_DIR, "pose_to_labelstudio_sdk.py")
        if os.path.exists(POSE_SCRIPT):
            logger.info(">>> 2/2 啟動 YOLO-Pose 骨架自動標註...")
            result_pose = subprocess.run(
                ["python", POSE_SCRIPT],
                capture_output=True,
                text=True,
                check=True
            )
            if result_pose.stdout:
                logger.info(f"YOLO-Pose 執行結果: {result_pose.stdout.strip()}")
        else:
            logger.warning(f"找不到 YOLO-Pose 標註腳本: {POSE_SCRIPT}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"同步腳本執行失敗 (Exit Code: {e.returncode})")
        logger.error(f"錯誤訊息: {e.stderr.strip()}")
    except Exception as e:
        logger.error(f"系統異常: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 監控服務已就緒，進入全自動模式...")
    
    while True:
        run_sync()
        logger.info(f"[*] 任務完成，進入休眠 {INTERVAL} 秒，系統維持待命狀態...")
        time.sleep(INTERVAL)



#「它是我們 MLOps 飛輪的『巡邏警衛與定時定向點火器』，負責每 5 分鐘自動把新照片抓下來，保證管線絕不罷工。」
#這是一個標準的運維與排程（Cron Job / Daemon）工具。
#我們前面在第十三關寫了一個超強的 inference_to_labelstudio_sdk.py，它負責把前線採集到 S3 的照片抓下來並自動做好標註、上傳到 Label Studio。但這支腳本本身是「被動」的，丟在那裡它自己不會動，必須有人去執行它。
#如果每次都要手動去跑，那就太不 MLOps 了。這個守護進程就是為了實現「完全無人值守」而設計的：
#五分鐘定時巡邏（Interval 300s）：
#程式啟動後會進入一個無限死循環（while True）。它就像一個不知疲倦的警衛，每隔 5 分鐘（300 秒）就會被喚醒一次，自動去調用第十三關的 SDK 同步腳本。
#多軌日誌紀錄（Logging System）：
#它配置了非常工業級的日誌系統。同步過程中噴出來的所有訊息、成功匯入了幾張圖、Webhook 有沒有點火成功，它會同時在終端機印出，並同步寫入本地的 watchdog.log 檔案。這樣一來，就算系統半夜崩潰，你隔天早上來翻日誌，也能一秒抓出是哪裡出問題。
#優雅退場機制（Signal Handling）：
#這段程式寫得非常具備系統工程師的素養！ 它手動引入了 signal 模組，監聽了系統的 SIGINT (鍵盤按下 Ctrl+C) 和 SIGTERM (系統關閉訊號)。當你想要關閉這個守護進程時，它不會直接暴力當機、在系統留下髒進程，而是會觸發 graceful_exit 函數，優雅、安全地印出一行「安全關閉監控服務」後才退場，保證作業系統的乾淨！