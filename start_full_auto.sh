#!/bin/bash

# =========================================================================
# 👑 MLOps 全自動學習閉環點火器 —— 終極極簡通車版 (畫面完美純淨體)
# =========================================================================

PROJECT_PATH="/Users/albert/Documents/專案/AIPE03/Fall"
TOOLS_PATH="$PROJECT_PATH/tools"
VENV_PATH="$PROJECT_PATH/.venv/bin/activate"

echo "🔄 正在強制清理背景舊進程與地端快取，確保演示最佳狀態..."

pkill -f watchdog.py
pkill -f webhook_receiver.py
pkill -f "clearml-agent"
kill -9 $(lsof -t -i:9001) 2>/dev/null

# rm -f $PROJECT_PATH/active_learning_dataset/images/*  # 註銷以啟用業界增量快取模式
# rm -f $PROJECT_PATH/active_learning_dataset/labels/*  # 註銷以啟用業界增量快取模式

echo "🚀 正在啟動全新的全自動 MLOps 閉環系統..."

source $VENV_PATH

# 🎯 🌟 [重要環境變數注入] 讀取地端 .env 憑證並導出，確保 Agent 咬單進程與 S3 權限對齊
if [ -f "$PROJECT_PATH/tools/.env" ]; then
    echo "🔑 已成功自 $PROJECT_PATH/tools/.env 導出最新的 AWS 與系統憑證環境變數！"
    export $(grep -v '^#' "$PROJECT_PATH/tools/.env" | xargs)
    export CLEARML_SDK__AWS__S3__KEY="$AWS_ACCESS_KEY_ID"
    export CLEARML_SDK__AWS__S3__SECRET="$AWS_SECRET_ACCESS_KEY"
    export CLEARML_SDK__AWS__S3__USE_CREDENTIALS_CHAIN="true"
fi

echo "[*] 正在啟動 Webhook 監聽器..."
nohup python $TOOLS_PATH/webhook_receiver.py --port 9091 > $PROJECT_PATH/webhook.log 2>&1 &

echo "[*] 正在啟動 S3 自動同步監控服務..."
nohup python $TOOLS_PATH/watchdog.py > $PROJECT_PATH/watchdog.log 2>&1 &

echo "[*] 正在啟動 ClearML Agent 異步咬單工人..."
nohup bash -c "PATH=\"$PROJECT_PATH/.venv/bin:\$PATH\" clearml-agent daemon --queue default --foreground" > $PROJECT_PATH/agent.log 2>&1 &

sleep 3

echo "------------------------------------------------"
echo "🔥 [常駐服務已就位] 立即點火「地端自動打標推理管道」..."
echo "------------------------------------------------"

# 🎯 所有的推論、打標、上傳、以及核心第十九關的全自動 API 同步，全部在這一支 Python 內一氣呵成！
python $TOOLS_PATH/inference_to_labelstudio_sdk.py

# 🚀 [全新點火] 預打標完成後，直接主動且精準地對 ClearML 發射「唯一一個」重訓任務！
echo "🔥 預打標完成，正在向 ClearML 佇列發射單一重訓任務..."
python $TOOLS_PATH/submit_task.py

echo "------------------------------------------------"
echo "💡 提示：此時您可以直接去刷新網頁，照片已經自動從 AWS 掉進 Label Studio 了！"
echo "你可以透過此指令查看背景重訓進度: tail -f $PROJECT_PATH/agent.log"
echo "------------------------------------------------"



#「它是我們整套系統的『一鍵還原與全線點火啟動總開關』。在評審和組長面前，按這一下，就能保證系統以 100% 完美的純淨狀態起跑！」
#在大型專案的現場 Demo 中，最怕的就是背景卡著上一次測試留下來的舊進程、或是硬碟裡留著昨天的髒照片快取，導致現場演示時檔案撞名、Port 被佔用而直接大翻車。
#這支 .sh 腳本就是你為了明天的演示現場，特別打造的「系統除錯與點火二合一雷達」：
#強制大掃除（pkill & rm）：
#一按下執行，它會像防毒軟體一樣，先用 pkill -f 幫你把背景偷跑的舊守護進程（watchdog.py）和舊中繼站（webhook_receiver.py）通通無情殺死，釋放被佔用的 Port 9000。接著，用 rm -f 抹除本地所有的相片與標註標籤快取，確保等一下新進來的跌倒照片是「100% 此時此刻新鮮採集」的！
#自動導流虛擬環境（source .venv）：
#自動幫你跑到專案路徑下，一鍵啟用 .venv 虛擬環境，完全不需要你在黑視窗裡手動打一長串路徑指令。
#背景悄悄點火與日誌雙分流（nohup & 🚀）：
#它會在背景用 nohup 的方式，同時把第八關/第十七關的中繼監聽器以及第十六關的 S3 守護進程安全地跑起來。並且把它們的輸出訊息分別導流到 webhook.log 和 watchdog.log。
#演示備忘指南（tail -f）：
#腳本最後還非常貼心地印出兩行提示。明天評審一來，你只要帥氣地在左邊視窗執行這支腳本，右邊視窗打 tail -f webhook.log，就能在螢幕上即時展示「 AI 正在數張數、數到 6 張立刻轟入雲端點火重訓」的震撼畫面！