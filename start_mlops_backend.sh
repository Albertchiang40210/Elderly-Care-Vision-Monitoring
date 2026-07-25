#!/bin/bash
echo "======================================================="
echo "🏋️ 啟動：後方 MLOps 資料標註與自動重訓後台 (M5 Pro)..."
echo "======================================================="

# 確保腳本執行時以腳本所在的最外層目錄為基準
cd "$(dirname "$0")"

# 1. 徹底清理上一次殘留的舊環境（強制 down 掉所有容器與殘留網路）
echo "🔌 正在強制清理舊有 Docker 運算服務..."
docker compose -f Fall/tools/docker-compose-label.yml down >/dev/null 2>&1
docker compose -f Fall/tools/docker-compose-triton.yml down >/dev/null 2>&1
docker compose -f Fall/tools/docker-compose-kafka.yml down >/dev/null 2>&1
docker compose -f Fall/tools/docker-compose-clearml.yml down >/dev/null 2>&1

# 2. 強制刪除任何可能殘留、沒有標籤的 tools_default 網路
docker network rm tools_default >/dev/null 2>&1

# 3. 🎯 關鍵：手動建立帶有正確 Compose Project 標籤的共享網路
# 這樣一來，後面所有的 docker-compose 都會認得它，絕對不會再報標籤錯誤！
echo "🌐 正在初始化 MLOps 內部共享網路..."
docker network create \
  --label "com.docker.compose.network=default" \
  --label "com.docker.compose.project=tools" \
  tools_default

# 4. 依序拉起服務
echo "🚀 正在啟動運算基礎設施..."
docker compose -f Fall/tools/docker-compose-clearml.yml up -d
docker compose -f Fall/tools/docker-compose-kafka.yml up -d
docker compose -f Fall/tools/docker-compose-triton.yml up -d

# 稍等 2 秒，拉起高度依賴 tools_default 網路的 Label Studio
sleep 2
docker compose -f Fall/tools/docker-compose-label.yml up -d

echo "⏳ 等待後台服務初始化 (8秒)..."
sleep 8

# 5. 啟動 ClearML 的背景自動訓練排程小精靈 (Agent)
echo "🤖 正在背景載入任務監聽 Agent Daemon..."
pkill -f "clearml-agent" >/dev/null 2>&1
nohup clearml-agent daemon --queue default >/dev/null 2>&1 &

echo "======================================================="
echo "✅ 後台管理中樞與 Triton 引擎部署完畢！"
echo "🌐 服務入口導覽："
echo "   - ClearML 後臺看板   : http://localhost:8080"
echo "   - Label Studio 標註   : http://localhost:8082"
echo "   - Triton gRPC 通道   : localhost:8001 (已就緒 🚀)"
echo "======================================================="
echo "💡 提示：此 VS Code 視窗已完全釋放！您可以直接在這個分頁繼續輸入其他指令。"


#「它是我們 MLOps 資料工廠的『總開關』。按這一下，後台的實驗看板（ClearML）、標註平台（Label Studio），以及『背景重訓小精靈（ClearML Agent）』會在一秒內全部各就各位！」
#在前線多鏡頭（Room 301, 302, 303）不斷採集模糊樣本、由 VLM 二審護理長打包成 JSON 的同時，後方必須要有一整套強大的「基礎設施」在隨時待命，接收這些資料並準備重訓。
#這支腳本扮演的就是後方資料工廠的點火總司令：
#基礎設施群雄歸位（Docker Compose）：
#它會自動跑到 Fall/tools/ 目錄下，把第六關寫的 ClearML 看板六大容器，以及第八關寫的 Label Studio 標註平台貨櫃 一鍵在背景（up -d）全部拉起來。並且貼心地在開頭先跑 down 來強制清理上一次殘留的舊環境，保證 100% 的純淨度！
#重訓小精靈無痕脫離（ClearML Agent Daemon）：
#這一動寫得極度有工業級 MLOps 運維老手的味道！
#光把看板開起來還不夠，必須要有一個「運算節點（Agent）」去盯著任務隊列（Queue）。你先用 pkill 清理掉舊的 Agent，接著用 nohup clearml-agent daemon --queue default >/dev/null 2>&1 &。
#這行神奇的指令會讓 Agent 真正脫離當前的終端機視窗，變成系統地下的背景守護進程（Daemon）。它會在暗中死死盯著任務隊列，只要第八關/第十七關的 Webhook 一點火，它就會在背景咬單、載入第十三關的 DEIM-DETR 密集優化結構圖紙，立刻開始進行模型進化重訓！
#完美釋放視窗與入口導覽：
#腳本最後用精美的排版印出了後台的所有入口網址（看板 8080、標註 8089），並提示你「此 VS Code 視窗已完全釋放」，這讓你可以在同一個終端機頁面直接去操作前線的指令，非常優雅！