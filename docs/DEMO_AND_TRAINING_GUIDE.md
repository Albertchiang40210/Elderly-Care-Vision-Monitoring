# Fall Detection - Demo 與重訓操作手冊

這份手冊整理 Demo 啟動方式與擴充資料重訓的指令。

---

## 場景一：Demo 啟動系統

依序開啟終端機分頁執行：

### 1. 喚醒雲端 MLOps 後勤指揮中心 (基礎設施)
這會啟動 Label Studio、ClearML 容器與 watchdog、webhook 監聽程式（資料庫與 Kafka 屬於基礎層，需另外 `docker compose up -d`）。
```bash
./start_cloud.sh
```

### 2. 啟動邊緣端攝影機與戰情室 (核心推論與網頁 UI)
請開啟另一個終端機分頁，這會啟動前端、後端 API、串流中繼站以及 AI 推論 Worker。
```bash
./start_edge.sh
```

### 3. 結束與關閉系統
當 Demo 結束時，只需執行以下指令即可安全關閉所有背景程序與 Docker 容器：
```bash
./stop_all.sh
```

---

## 🧠 場景二：平時的 MLOps 自動重訓 (讓 AI 變聰明)

三個模型各自有獨立的重訓腳本，可以只重訓其中一個。
**注意：以下所有指令，都請在專案根目錄 (`AIPE03`) 執行！**

### 🟢 模型 A：RT-DETR (負責抓人體邊界框)
**何時需要練它？** 當您發現 AI 常常抓不到畫面邊緣的人、或是誤把假人當真人時。
1. **餵資料**：把新的照片丟進 `Fall/label_studio_data/images`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/inference_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio (`http://localhost:8082`) 調整邊界框，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_train_pipeline.py
   ```

### 🟡 模型 B：YOLO-Pose (負責抓 17 個關節)
**何時需要練它？** 當您發現 AI 框到人了，但骨架亂飄、腳跟頭反過來時。
1. **餵資料**：把新的照片丟進 `Fall/label_studio_data/pose_images`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/pose_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio 微調錯位的關節點，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_pose_train_pipeline.py
   ```

### 🟣 模型 C：Action Transformer (負責判斷連續動作)
**何時需要練它？** 當骨架都很準，但 AI 卻把「安全蹲下(squat)」誤判成「危險跌倒(fall)」時。
1. **餵資料**：把短短幾秒的動作影片丟進 `Fall/label_studio_data/videos`。
2. **AI 幫您寫作業 (預標註)**：
   ```bash
   python Fall/tools/action_to_labelstudio_sdk.py
   ```
3. **人工批改 (審核)**：去 Label Studio 選擇正確的動作標籤 (fall / sitdown / squat 等)，按下 Submit。
4. **送進煉丹爐 (自動重訓 + 熱部署)**：
   ```bash
   python Fall/tools/clearml_action_train_pipeline.py
   ```

---
💡 **注意**：推論 Worker (`inference_worker.py`) 在啟動時載入當時最新的權重；重訓完成後需**重新啟動 Worker** 才會使用新模型（Triton 設定為輪詢模式，若走 Triton 部署則會自動載入新版本，但我沒有驗證過這條路徑）。
