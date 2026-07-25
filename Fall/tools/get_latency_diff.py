import json
import time
from kafka import KafkaConsumer

# 監聽你的雙軌 Topic
# 快速道路: processed-reports / 慢速道路: nursing-home-alerts (或經 vlm 處理後的 topic)
consumer = KafkaConsumer(
    'processed-reports', 'nursing-home-alerts',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("📊 [秒數差觀測站] 正在監聽 MLOps 管線數據...請觸發跌倒警報...")
print("================================================================")

try:
    for message in consumer:
        payload = message.value
        topic = message.topic
        arrival_time = time.time()  # 護理站接收到資料的當下時間
        
        edge_time = payload.get("edge_detect_time")
        alert_id = payload.get("alert_id", "Unknown")
        alert_type = payload.get("alert_type", "Unknown")
        
        if not edge_time:
            print(f"⚠️ 收到來自 {topic} 的封包，但裡面沒有埋 edge_detect_time 時間戳記，無法計算！")
            continue
            
        # 計算端到端總延遲（秒）
        total_latency_seconds = arrival_time - edge_time
        
        print(f"\n🔔 [擷取到警報事件] ID: {alert_id}")
        print(f"   📂 走哪條路: {'🧠 慢速道路 (VLM二審)' if 'VLM' in alert_type else '⚡ 快速道路 (純YOLO直發)'}")
        print(f"   ⏱️ 邊緣發出時間: {time.strftime('%H:%M:%S', time.localtime(edge_time))}.{int((edge_time%1)*1000):03d}")
        print(f"   ⏱️ 護理站收到時間: {time.strftime('%H:%M:%S', time.localtime(arrival_time))}.{int((arrival_time%1)*1000):03d}")
        print(f"   🚀 【總共花費時間】: {total_latency_seconds:.4f} 秒")
        print("----------------------------------------------------------------")

except KeyboardInterrupt:
    print("\n觀測結束。")



#「它是我們智慧病房系統的『秒錶與道路流量儀』，用來證明我們的警報傳得有多快！」
#在智慧病房的臨床情境中，「警報延遲時間」是決定這個系統能不能上線的死穴。如果病患跌倒了，系統過了 10 秒才通報，那病患可能已經發生二次傷害。
#這個檔案扮演的就是護理站的接收端觀測台：
#雙軌通道監聽：它在背景同時監聽 Kafka 的兩個收發通道：
#⚡ 快速道路（processed-reports）：前線 YOLOv11-pose 偵測到跌倒後，完全不廢話，直接秒發的警報。
#🧠 慢速道路（nursing-home-alerts）：需要經過大語言模型（VLM / 多模態模型）二次審查，確認是不是誤判後才發出的警報。
#精準計算延遲：當它一抓到警報封包，就會把「護理站收到時間」減去「邊緣端發出時間（edge_detect_time）」，算出端到端的精確秒數差，精度高達小數點後四位（毫秒級）！
#數據視覺化呈現：它會在終端機用非常漂亮的排版，即時印出每筆警報走了哪條路、花了幾秒。