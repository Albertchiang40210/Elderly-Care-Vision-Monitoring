# 🏥 智慧病房主動學習 MLOps 即時防跌落管線系統 (Smart Ward Active Learning MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Framework](<https://img.shields.io/badge/Framework-FastAPI%20%7C%20Ultralytics%20%7C%20Ollama-green.svg>)
![Pipeline](<https://img.shields.io/badge/Pipeline-NVIDIA%20DeepStream%207.0%20%7C%20Triton%20Server-76B900.svg>)
![MLOps](<https://img.shields.io/badge/MLOps-ClearML%20%7C%20Label%20Studio-orange.svg>)
![Infrastructure](<https://img.shields.io/badge/Infrastructure-Docker%20%7C%20Kafka%20%7C%20MediaMTX%20%7C%20AWS%20S3-lightgrey.svg>)

> **工業級智慧醫療監控落地解決方案**：結合前線多鏡頭時序推理邊緣端（NVIDIA DeepStream 7.0 / Python 雙模式引擎）、Triton Inference Server、後台大語言模型（Video-LLM）雙軌二審機制、與 Label Studio 數據閉環，全面打通「異常採集 ➔ 雲端下沉 ➔ 智慧預標註 ➔ 非阻塞異步點火 ➔ 背景 Agent 密集監督增量重訓」的全自動 MLOps 主動學習飛輪。

---

## 🏗️ 核心檔案與架構全景導覽 (Core Components)

本專案由關鍵模組、腳本、Docker 容器與組態配置交織而成，以下為各檔案在系統中的「白話文使命與核心職責」：

### 📁 第一階段：環境防禦與資料基石 (Env & Data Basics)

* **1. `.env`（專案總帳本）**
  * *白話使命*：全系統的安全核心，集中管理 AWS 雲端密鑰（S3）、RDS 資料庫帳密、以及 Kafka 與 Label Studio 的連線 Port，拒絕將密鑰硬編碼在程式中。
* **2. `.gitignore`（隱私過濾網）**
  * *白話使命*：專案上傳 GitHub 時的隱形過濾器。強制阻擋動輒數百 MB 的大模型權重檔（`.pt` / `.pth`）、有機密帳密的 `.env`、資料庫憑證 `.pem` 以及 `backend/` 資料夾，防範公司資產被看光。
* **3. `data.yaml`（模型課本目錄）**
  * *白話使命*：後台大腦增量訓練時的「目錄索引表」。使用絕對路徑精準映射地端的資料夾（`images/` 和 `labels/`），明定重訓類別（`person`, `bed`, `wheelchair`），防止 Ultralytics 找不到課本而罷工。
* **4. `clean_pose_to_det.py`（特徵資料清洗器）**
  * *白話使命*：主動學習的清洗工人。前線採集到的長輩跌倒姿勢往往夾雜雜訊，本腳本專職過濾掉骨架關節不完整的髒數據，將其提煉並轉換成標準 YOLO 方形框座標。

### 📁 第二階段：前線多模態邊緣推論與雙模式 AI 引擎 (Edge Inference & Dual-Mode Pipeline)

* **5. `deepstream_configs/deepstream_app_config.txt`（DeepStream 7.0 產線級 Pipeline 配置文件）**
  * *白話使命*：NVIDIA 高效能 GPU 推理管線的組態大腦。定義了原始 RTSP 接入、`nvinferserver` (Triton Server)、`nvtracker` 多目標追蹤、OSD 骨架與跌倒框即時繪製、MediaMTX (RTSP/WebRTC) 串流輸出與 Kafka 告警傳播器 (`nvmsgbroker`)，實現全 GPU 零拷貝（Zero-Copy）毫秒級處理。
* **6. `action_transformer.pth`（時序動作記憶體）**
  * *白話使命*：存放 ActionTransformer 模型的微調核心權重，負責賦予系統「觀察連續 30 幀動作」的時序記憶能力。
* **7. `inference_test.py`（前線多鏡頭巡邏 Worker - Dev 模式）**
  * *白話使命*：智慧病房第一線的 Python AI 巡邏警衛。在 Mac/CPU 本地開發環境下，利用 GStreamer/OpenCV 多線程平行拉取攝影機畫面，融合 YOLO11-Pose 與 ActionTransformer；內建「跌倒自行站起姿態自動恢復 (Self-Recovery)」與「背景多執行緒 10 秒影片合成傳 S3」無感降維技術，實現地端無痕隱私清理與高流暢度監控。
* **8. `get_latency_diff.py`（秒數差觀測站）**
  * *白話使命*：護理站的流量儀與秒錶。同時監聽 Kafka 的快速道路與二審佇列，精確算出警報從邊緣端發出到護理站收到的「端到端總延遲時間」，精度高達毫秒級，用直觀的秒數差向評審證明系統的高吞吐量。

### 📁 第三階段：雲端資安與 Docker 基礎建設 (Infra & DevOps)

* **9. `global-bundle.pem`（AWS RDS 加密連線數位印章）**
  * *白話使命*：工業級資安防火牆。裡面是 AWS 官方核發的數位憑證總集合，Python 程式在連線 RDS 資料庫時會強制開啟 SSL/TLS 加密，與這枚印章核對，確認對方是 AWS 官方資料庫而非山寨網站，保障病患隱私。
* **10. `docker-compose-clearml.yml`（實驗重訓看板六大貨櫃圖紙）**
  * *白話使命*：MLOps 重訓後台的一鍵建造圖紙。自動將 Redis、MongoDB、Elasticsearch、APIServer、Web控制台（Port 8080）與 FileServer 打包聯網，提供完美的實驗追蹤後台。
* **11. `docker-compose-kafka.yml`（訊息公路與地端資料庫圖紙）**
  * *白話使命*：整套串流系統的交通樞紐。一鍵開啟 Zookeeper、Kafka（Port 9092）訊息收發中心，以及用於儲存病患資料與告警歷史的 PostgreSQL（Port 5433）中央倉庫。
* **12. `docker-compose-label.yml`（標註平台控制台圖紙）**
  * *白話使命*：主動學習的人工審核控制台。在背景一鍵跑起 Label Studio 容器（Port 8089），並掛載實體磁碟卷防止標註好的黃金教材遺失。

### 📁 第四階段：後台大腦二審與主動學習閉環 (VLM & Active Learning)

* **13. `vlm_worker.py`（護理長大腦：VLM 雙軌二審監聽器）**
  * *白話使命*：護理站的 AI 總護理長。負責監聽 Kafka 警報並執行三區間分流策略。當置信度模糊時，自動喚醒本地端 `Qwen2.5-VL` 進行原生影片時序二審。審核成功後，自動在本地 `predictions/` 生成 Label Studio Standards 預測 JSON，解鎖主動學習閉環。
* **14. `inference_to_labelstudio_sdk.py`（AI 自動預標註與 S3 下沉同步器）**
  * *白話使命*：後台的 AI 自動畫框外包小幫手。模擬瀏覽器登入標註平台，若照片處在雲端 S3 則透過 Boto3 串流下載到地端（對齊 ClearML重訓目錄），並利用大腦模型進行環境偵測（`bed`, `chair` 等），直接以「已正式提交」的狀態灌回網頁。
* **15. `webhook_receiver.py`（自動化計數點火閥）**
  * *白話使命*：全自動點火控制閥。在 Port 9001 默默數著 Label Studio 傳回來的 Webhook 張數。當累積標註滿設定張數時，利用 asyncio 異步非阻塞技術與 `asyncio.Lock()` 併發鎖防禦，彈射發射重訓任務，完全不卡網頁主水管與競態衝突。
* **16. `watchdog.py`（輪詢掃描監控守護進程）**
  * *白話使命*：不知疲倦的維運警衛。每隔 5 分鐘（300 秒）自動被喚醒，去驅動同步與自動標註腳本，搭配多軌 `watchdog.log` 日誌，保證主動學習管線 24 小時不斷線。

### 📁 第五階段：雲端增量重訓與結構優化 (Incremental Training)

* **17. `rtdetr-l-deim.yaml`（DEIM 密集輔助監督結構藍圖）**
  * *白話使命*：自研大腦的獨家武功秘笈。精簡官方參數，在訓練時注入「一對多密集匹配（Dense Matching）」大幅提升對棉被遮擋、輪椅欄杆等複雜病房環境的學習精準度；推論時自動收回精簡，完全不拖慢前線毫秒級的速度。
* **18. `clearml_train_pipeline.py`（雲端增量重訓核心大腦）**
  * *白話使命*：真正執行模型進化的重訓大腦。自動去抓取剛剛預標註並審核完的 6 張新鮮教材，執行增量訓練，並在重訓結束後將新權重自動推回 AWS S3 模型倉庫。
* **19. `submit_task.py`（總指揮官點火腳本）**
  * *白話使命*：重訓飛輪的避雷針與點火器。在 ClearML 看板建立任務殼，注入 `CLEARML_DISABLE_GIT_DETECTION` 變數強制繞過 Git 檢查防止崩潰；自動外掛雲端「自力救濟安裝套件碼」，並利用 `subprocess` 開闢原生進程直接開訓，防止線程死鎖。

### 📁 第六階段：大自動化一鍵通電運維 (DevOps Automations)

* **20. `start_all.sh`（一鍵總點火主控制腳本 - 智慧雙模式自動偵測）**
  * *白話使命*：全系統 Master 啟動開關。內建 **NVIDIA GPU 智慧硬體感測探針**：在有 N 卡環境自動切換至 **NVIDIA DeepStream 7.0 容器 Pipeline (Prod 模式)**；在 Mac/無 N 卡環境自動切換至 **Python + OpenCV 巡邏引擎 (Dev 模式)**。自動啟動 Docker 基礎設施、Kafka 消費者、MediaMTX 串流中繼與全自動 MLOps 監視服務。
* **21. `start_full_auto.sh`（前線微服務合流一鍵大通電腳本）**
  * *白話使命*：中央配電盤。自動解除 Conda 環境對 `PATH` 的污染。一鍵開闢背景 Docker、點活夥伴 Kelly 的 FastAPI 後端（Port 8000）與 Consumer，切進 Fall 環境開啟 VLM 大腦。

---

## 🛠️ 環境依賴配置 (Requirements)

專案虛擬環境完全體套件清單如下（已全面過濾冗餘相依，鎖定最穩定工業級版本）：

```text
python-dotenv             # 用於讀取 .env 檔案中的雲端憑證、資料庫與 Kafka 設定
requests                  # 用於模擬瀏覽器登入與呼叫 Label Studio API 自動注入標註
torch                     # PyTorch 核心深度學習框架 (用於 ActionTransformer 與張量序列運算)
torchvision               # PyTorch 影像處理擴充套件
ultralytics>=8.1.0        # YOLOv11 & RT-DETR/DEIM-DETR 核心推論與增量重訓套件
numpy                     # 基礎矩陣運算與邊緣端影像資料流處理
ollama                    # 本地端大模型驅動 SDK (用於調用 Qwen2.5-VL 進行影片二審判讀)
langgraph                 # Agent 狀態圖編譯與事件生命週期狀態機框架 (用於 vlm_worker 護理長大腦)
langchain-core            # LangGraph 基礎元件與 Prompt/Message 封裝
boto3                     # AWS 官方 Python SDK (解鎖邊緣端直傳 S3 與自動化下載下沉對齊)
clearml[s3]>=1.14.0       # MLOps 實驗追蹤核心，整合 [s3] 擴充以支援自動同步重訓模型至雲端
psycopg2-binary           # 用於安全連接 AWS RDS (PostgreSQL) 關聯式資料庫
kafka-python-ng           # 用於讀取與寫入 Kafka 訊息佇列 (相容 Python 3.12+)
sqlalchemy                # ORM 框架，用於對關聯式資料庫進行物件關係對應操作
python-jose[cryptography] # JWT Token 簽名與解密驗證工具
passlib[bcrypt]           # 密碼雜湊與安全比對工具
python-multipart          # 允許 FastAPI 解析前端登入表單資料
fastapi                   # 用於建立高吞吐量的 Webhook 監聽服務，對接 Label Studio 事件
uvicorn                   # FastAPI 的高效能 ASGI Web 伺服器引擎
opencv-python             # OpenCV 核心影像與影片處理套件 (用於邊緣端影像流)
