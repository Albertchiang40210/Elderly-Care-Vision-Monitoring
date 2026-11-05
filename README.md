# 長照機構跌倒偵測與主動學習系統

長照機構的跌倒偵測系統。用 YOLO11-Pose 取得人體骨架，再用自己訓練的 Transformer 判斷連續 30 幀的動作；信心不高的事件交給本地 VLM（Qwen2.5-VL）二次審核。事件由 FastAPI 後端接收、透過 SSE 推送到 React 護理站儀表板，並把誤報與低信心樣本送進 Label Studio / ClearML 做標註與重訓。

> 這是個人專案，在筆電與一台 GCP VM 上開發與展示，**尚未在真實機構部署，也沒有經過長期穩定性或壓力測試**。下方「已知限制」有完整說明，數據都附上來源與測量條件。

## 系統架構

```
RTSP 攝影機 ─► MediaMTX / go2rtc ─► 推論 Worker (Fall/tools/inference_worker.py)
                                       │  YOLO11-Pose → 骨架序列(30 幀×34 維) → ActionTransformer
                                       │  (支援直接載入模型，或透過 Triton Inference Server 呼叫 ONNX)
                                       ▼
                                  Kafka 事件
                       ┌──────────────┴───────────────┐
              信心高：直接送出                  信心中：vlm_worker.py
                       │                     (LangGraph + Ollama/Qwen2.5-VL 看影片二審)
                       └──────────────┬───────────────┘
                                      ▼
                  backend/kafka_consumer.py ─► POST /events (X-API-Key)
                                      ▼
              FastAPI + PostgreSQL ──SSE──► React 護理站儀表板
                    │ 截圖存本機 backend/static/images（以 /static 提供）
                    ▼
        誤報 / 低信心樣本 ─► Label Studio 標註 ─► Webhook 計數 ─► ClearML 重訓 ─► 新權重存本機/ClearML
```

### 實作了什麼、沒實作什麼

| 項目 | 狀態 |
|---|---|
| 骨架 + Transformer 動作分類（7 類） | 已實作並訓練 |
| 規則式跌倒判斷（身體角度、EMA 追蹤）`Fall/modules/fall_detector.py` | 已實作 |
| VLM 二審（LangGraph 分流，三區間門檻） | 已實作，**效能尚未量化評估** |
| FastAPI 後端（JWT 登入、事件、裝置、報告、SSE） | 已實作，有自動化測試 |
| React 儀表板（20 個頁面） | 已實作，無前端自動化測試 |
| Triton 部署 ONNX 模型 | 設定檔已提供，模型檔不在 repo（見下） |
| DeepStream 7.0 | 只有一份設定檔 `deepstream_configs/deepstream_app_config.txt`，**沒有在 NVIDIA GPU 上實際跑過**；實際展示用的是 Python 推論 Worker |
| 自動重訓流程（Webhook → ClearML） | 腳本已寫，**沒有驗證過重訓後的模型比舊模型更好** |
| RT-DETR 的 DEIM 設定 `rtdetr-l-deim.yaml` | 有設定檔，**沒有對照實驗**，不宣稱準確度提升 |

## 測試與數據

以下數據都是我實際執行或從 repo 檔案計算得到的。

### 後端自動化測試（pytest）

執行：`cd backend && PYTHONPATH=..:. uv run pytest -q`（測試使用 in-memory SQLite，不需要啟動 Postgres）

| 項目 | 結果 |
|---|---|
| 測試檔數 | 23 個 |
| 結果 | **120 個全部通過**（約 54 秒） |
| 程式碼覆蓋率 | **約 78%**（787 行應用程式碼中 616 行被執行；已排除測試檔與一次性腳本 `create_seed_data.py`、`init_db.py` 等） |

覆蓋最弱的是 `events/router.py`（71%）與 `kafka_consumer.py`（59%，`run()` 主迴圈需要真的 Kafka，沒有測）。測試只涵蓋後端，前端沒有自動化測試。

### 動作分類模型（ActionTransformer）

| 項目 | 數值 |
|---|---|
| 輸入 | 30 幀 × 17 個關鍵點 × (x, y) = 30×34 |
| 模型 | 2 層 Transformer encoder，d_model=64，4 heads |
| 類別 | bend、fall、kneel、normal、sitdown、squat、walk（7 類） |
| 資料集 | 19,392 筆序列（約 75 段連續片段） |

各類別筆數：fall 9,524 ／ walk 2,524 ／ sitdown 1,664 ／ squat 1,640 ／ kneel 1,480 ／ bend 1,316 ／ normal 1,244

**兩種切分方式的結果差很多：**

| 驗證方式 | 驗證準確率 |
|---|---|
| 隨機切 80/20（訓練腳本 `train_action_classifier.py` 的做法，`metrics.json` 的 97.68%） | 97.68%（取驗證集上最佳的 epoch） |
| **依連續片段切 80/20**（`Fall/tools/eval_grouped_split.py`，30 epochs，最後一個 epoch） | **66.7%** |

原因：序列是用滑動窗口（stride 5、長度 30）從連續影片切出來的，相鄰窗口高度重疊。隨機切分會讓幾乎相同的樣本同時出現在訓練與驗證集，**97.68% 嚴重高估**真實泛化能力。依片段切分後，整片段都沒看過，才比較接近換新場景時的表現。另外 `best_acc` 是在驗證集上挑最佳 epoch，也有些許樂觀偏差。

依片段切分的分類報告（驗證 2,520 筆、15 段）：

| 類別 | precision | recall |
|---|---|---|
| fall | 0.664 | **0.932** |
| normal | 0.789 | 0.956 |
| walk | 0.657 | 0.665 |
| sitdown | 0.548 | 0.611 |
| kneel | 0.703 | 0.531 |
| bend | 0.286 | 0.016 |
| squat | 0.237 | 0.046 |

跌倒的 recall 偏高（0.93）但 precision 只有 0.66，多數誤判來自其他動作被判成 fall；bend、squat 幾乎認不出來。這也是系統需要規則式判斷與 VLM 二審來降低誤報的原因。

**這個評估本身的限制：**
- 資料集沒有影片編號，我用「連續相同標籤」近似「同一支影片」，可能把同一支影片切成多段，或相反；只有 15 段驗證片段，變異很大。
- 只訓練 30 epochs（正式訓練是 150），單一隨機種子，沒有重複實驗。
- 資料來源是展示用與自行錄製的場景，不是真實長者。
- 因此 66.7% 是粗略的下界式參考，不是穩定的指標。要得到可靠數字，需要依影片重新蒐集並切分資料。

### 資料集規模

| 資料 | 數量 |
|---|---|
| 物件偵測影像（`active_learning_dataset/images`） | 1,376 張 |
| 動作特徵序列 | 19,392 筆 |

### 後端 API 處理時間（僅後端，非端到端）

`BENCH=1 PYTHONPATH=..:. uv run pytest -s tests/test_bench_events.py`（在 `backend/` 下）

| 項目 | 結果 |
|---|---|
| 量測內容 | `POST /events` 300 次，含驗證、寫入 in-memory SQLite |
| 環境 | Apple M5 Pro（arm64），TestClient，無網路 |
| 中位數 / P95 / 最大 | 1.7 ms / 1.9 ms / 9.1 ms |

這只代表後端處理邏輯本身很快；**不含** Kafka、網路、PostgreSQL 與 AI 推論，不能當作端到端延遲。

### 尚未量測的項目

下列項目我**沒有可信的測量值**，所以不寫數字：
- 端到端警報延遲（需要完整 Kafka + 推論環境，`Fall/tools/get_latency_diff.py` 可量測，但我沒有保存過正式結果）
- 跌倒偵測的 Recall / Precision，以及加入 VLM 前後的差異
- 多路攝影機同時推論的上限
- 重訓前後模型的準確率變化

> 先前版本的簡報圖表腳本裡有一組寫死的 VLM 前後 Recall/Precision 數字（非實測），已連同圖檔刪除。目前 repo 內沒有任何 VLM 效能數據。

## VLM 二審的分流邏輯

`Fall/tools/vlm_worker.py` 用 LangGraph 做狀態圖，依事件型態與信心分數分流：

- 事件型態為 `fall`（Kafka 上的跌倒事件本來就是偏低信心才會送來）或例行環境檢查 → 一律進 VLM
- 其他事件：分數 ≥ 0.75 直接放行；0.35 ≤ 分數 < 0.75 進 VLM；< 0.35 丟棄
- VLM 使用本地 Ollama 的 `qwen2.5vl:latest`，審核結果可產生 Label Studio 預測 JSON 供人工確認

這些門檻是我憑經驗設定的，沒有用資料做過校準。

## 後端 API（FastAPI）

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/login`、`/register` | JWT 登入；註冊需 admin |
| GET/PATCH | `/me`、`/me/password` | 個人資料與改密碼 |
| GET/PATCH/DELETE | `/users`、`/users/{id}` | 使用者管理（admin） |
| GET/PATCH | `/devices`、`/devices/{id}` | 攝影機裝置 |
| POST | `/events` | 判斷層送事件，以 `X-API-Key` 驗證（機器對機器） |
| GET | `/events` | 事件列表 |
| GET | `/events/{id}/media` | 若 `clip_path`/`snapshot_path` 是 `s3://` 才回 presigned URL，否則回 null（見下方說明） |
| POST | `/events/{id}/ack` | 護理人員確認收到 |
| PATCH | `/events/{id}/resolve`、`/verdict` | 結案、判定真/誤報 |
| GET/POST | `/events/{id}/reports` | 事件報告 |
| GET | `/events/live-detection/stream` | SSE 即時推送 |

## 儲存與雲端依賴

整個系統在本機（Mac / 地端主機）以 Docker Compose 與本機程序執行：PostgreSQL、Kafka、Label Studio、ClearML 都是自架容器，VLM 使用本機 Ollama。截圖寫在 `backend/static/images`，由後端以 `/static` 提供。

程式碼裡保留了**選用的** S3 整合：`backend/core/s3.py` 會把 `s3://bucket/key` 換成限時網址，`vlm_worker.py` 也能從 S3 讀取。這些在地端展示時**沒有啟用**，也沒有針對真實 S3 做過測試（單元測試用 mock）。`GET /events/{id}/media` 對非 `s3://` 的路徑會回傳 null，地端模式下推論 Worker 寫入的 `clip_path`/`snapshot_path` 是本機相對路徑（例如 `/images/...`），不經過這個端點。

## 技術棧

- **AI**：PyTorch、Ultralytics（YOLO11-Pose、RT-DETR）、ONNX Runtime、Triton Inference Server、Ollama（Qwen2.5-VL）、LangGraph
- **後端**：FastAPI、SQLAlchemy、PostgreSQL（測試用 SQLite）、Kafka（kafka-python-ng）、JWT（python-jose）、boto3
- **前端**：React 19、TypeScript、Vite、Tailwind CSS 4
- **MLOps**：Label Studio、ClearML
- **基礎設施**：Docker Compose（profiles）、MediaMTX、go2rtc（本專案**全部地端執行，不依賴 AWS 或其他雲端服務**）

## 目錄結構

```
backend/            FastAPI 後端（core/、events/、devices/、users/、reports/、tests/）
frontend/           React 儀表板
Fall/modules/       骨架序列追蹤、ActionTransformer、規則式跌倒判斷
Fall/tools/         推論 Worker、VLM worker、標註同步、ClearML 訓練與部署腳本
Fall/model_repository/  Triton 模型設定（config.pbtxt）
deepstream_configs/ DeepStream 設定檔（未實測）
gcp_vm_environment/ 另一組 Kafka / Nginx / Prometheus 部署設定（VM 部署用，展示環境未使用）
RTSP.MediaMTX_20260723/  MediaMTX 設定與排查筆記
docker/             額外的 Dockerfile（webhook receiver）
docs/               MLOps 操作文件（ClearML、Label Studio、重訓 SOP、Demo 手冊）
scripts/            維運與一次性腳本
docker-compose.yml  以 profiles 分成 infra / edge / cloud
```

依賴管理：後端以 `backend/pyproject.toml` + `backend/uv.lock` 為準；`requirements.txt`（根目錄）是 `Fall/` 下 AI 與 MLOps 腳本的依賴。

## 執行方式

```bash
cp .env.example .env              # 填入資料庫、SECRET_KEY、EVENT_API_KEY；S3 相關欄位可留空

docker compose up -d              # 基礎層：PostgreSQL、Zookeeper、Kafka
docker compose --profile edge up -d    # 後端、前端、Triton、Netdata
docker compose --profile cloud up -d   # Label Studio、ClearML、Webhook receiver
```

也可使用 `start_edge.sh` / `start_cloud.sh` / `stop_all.sh` 在本機啟動各服務。

**取得模型權重（repo 不含）**
1. 準備資料並訓練：`Fall/tools/extract_action_features.py` → `Fall/tools/train_action_classifier.py`
2. 匯出 ONNX：`Fall/tools/export_action_model.py`（輸出到 `Fall/model_repository/action_transformer/<版本>/model.onnx`）
3. 姿態與偵測模型由 `Fall/tools/export_all_to_onnx.py` 匯出；`yolo_pose`、`rt_detr` 需放到對應的 `model_repository` 目錄

**注意**
- 模型權重（`.pt`、`.onnx`）因檔案大小與 `.gitignore` 設定不在 repo 裡，需自行訓練或匯出（`Fall/tools/export_all_to_onnx.py`），並放到 Triton 的 model repository。
- VLM 需要本機先安裝 Ollama 並 `ollama pull qwen2.5vl`。
- `docker-compose.yml` 內的部分預設值（資料庫帳密、掛載路徑）是開發用，正式使用前請改掉。

## 持續整合

`.github/workflows/ci.yml` 會在 push / PR 時執行後端 pytest 與前端 `npm run build`。前端 `npm run lint` 目前有 4 個既有錯誤（`no-explicit-any` 等），CI 內設為不阻擋。CI 設定我只在本機驗證過各步驟指令，**尚未在 GitHub 上實際跑過**。

## 已知限制

- 隨機切分的 97.68% 有資料洩漏；依片段切分只有約 66.7%，目前沒有在獨立錄製的資料上驗證過。bend、squat 幾乎辨識不出來。
- 訓練與展示資料不是真實長照機構的資料。
- 沒有量測延遲與併發能力，沒有負載測試。
- DeepStream 未實測；Triton 的 `action_transformer` 輸出維度已修正為 7 類，但我沒有實際用 Triton 載入並推論驗證過。
- Dockerfile 的 backend 與 webhook 映像我沒有實際 build 驗證過（撰寫時 Docker 未啟動），compose 檔語法已用 `docker compose config` 檢查。
- 推論 Worker 在啟動時載入最新權重，重訓後需重啟才會使用新模型。
- 前端沒有自動化測試。
- 安全性：compose 內有開發用預設資料庫帳密；CORS 預設只允許本機來源，部署時需設定 `ALLOWED_ORIGINS`。`EVENT_API_KEY` 必須自行在 `.env` 設定。

## 下一步（Roadmap）

1. 依影片/人物重新切分資料集並補資料，優先改善 bend、squat，重新評估動作分類模型。
2. 用標註好的測試影片量測 Recall / Precision，並比較有無 VLM 二審的差異。
3. 在完整環境量測端到端延遲，保存結果。
4. 驗證 Triton 部署路徑（ONNX 匯出 → 載入 → 推論結果與 PyTorch 一致）。
5. 修正前端 lint 錯誤並補上測試；後端補 `kafka_consumer.run()` 的整合測試。

## 我從這個專案學到的

- 把 AI 推論結果接進一個完整的事件系統（Kafka → API → SSE → 前端）要處理的重試、冪等與驗證問題。
- 用「規則 + 時序模型 + VLM」分層降低誤報成本的設計取捨。
- 資料切分方式對指標可信度的影響：同一份資料，隨機切 97.7%、依片段切 66.7%。下一步是依影片/人物重新切分並補資料，尤其是 bend 與 squat。
