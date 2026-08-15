# kafka_consumer.py
# Kafka consumer：消費 processed-reports，每則轉打 POST /events。
# 優化版：使用批次拉取 (poll) + 多執行緒 (ThreadPoolExecutor) + HTTP Keep-Alive 加速發送。

import json
import logging
import time
import traceback
import concurrent.futures

import httpx

try:
    from backend.core.config import (
        KAFKA_BOOTSTRAP_SERVERS,
        KAFKA_TOPIC,
        KAFKA_GROUP_ID,
        EVENTS_URL,
        EVENT_API_KEY,
        RETRY_SLEEP_SECONDS,
    )
except ModuleNotFoundError:
    from core.config import (
        KAFKA_BOOTSTRAP_SERVERS,
        KAFKA_TOPIC,
        KAFKA_GROUP_ID,
        EVENTS_URL,
        EVENT_API_KEY,
        RETRY_SLEEP_SECONDS,
    )

logger = logging.getLogger("kafka_consumer")


def classify_response(status_code: int) -> str:
    # 201 建立成功；400/422 是毒訊息（重試無用，跳過）；其餘（5xx/未知）當一時失敗重試
    if status_code in (200, 201):  # 👈 相容 200/201 成功狀態碼
        return "ok"
    if status_code in (400, 422):
        return "poison"
    return "retry"


def process_message_with_retry(msg_value: bytes, client: httpx.Client) -> str:
    """處理單筆 Kafka 訊息，包含解析、發送與失敗重試邏輯（執行於 ThreadPool 內）"""
    while True:
        try:
            data = json.loads(msg_value)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            logger.error("JSON 解析失敗 (毒訊息)，跳過：%r", msg_value)
            return "poison"
            
        try:
            response = client.post(
                EVENTS_URL,
                json=data,
                headers={"X-API-Key": EVENT_API_KEY},
                timeout=10,
            )
            decision = classify_response(response.status_code)
        except Exception:
            traceback.print_exc()
            decision = "retry"
            
        if decision == "retry":
            logger.warning("送出失敗（一時），%s 秒後重試", RETRY_SLEEP_SECONDS)
            time.sleep(RETRY_SLEEP_SECONDS)
            continue
        elif decision == "poison":
            logger.error("API 回傳 400/422 (毒訊息)，跳過：%r", msg_value)
            return "poison"
        else:
            return "ok"


def build_consumer():
    from kafka import KafkaConsumer

    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
        group_id=KAFKA_GROUP_ID,
        enable_auto_commit=False,   # 處理成功才手動 commit（at-least-once）
        auto_offset_reset="latest",  # 首次啟動只收「從現在開始」的新警報
    )


def run():
    consumer = build_consumer()
    logger.info("consumer 啟動，監聽 topic=%s bootstrap=%s", KAFKA_TOPIC, KAFKA_BOOTSTRAP_SERVERS)
    
    # 建立持久化的 httpx.Client (Keep-Alive) 以及 ThreadPoolExecutor 進行多執行緒並發
    with httpx.Client() as client, concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        try:
            while True:
                # 批次拉取訊息 (最多等待 1 秒)
                msg_pack = consumer.poll(timeout_ms=1000, max_records=100)
                if not msg_pack:
                    continue
                
                # 展開各 Partition 的訊息
                all_messages = []
                for tp, messages in msg_pack.items():
                    all_messages.extend(messages)
                    
                if not all_messages:
                    continue
                    
                # 將所有訊息丟入 ThreadPool 平行處理
                futures = []
                for msg in all_messages:
                    futures.append(executor.submit(process_message_with_retry, msg.value, client))
                    
                # 阻塞直到這批訊息全數處理完畢 (ok 或 poison)
                concurrent.futures.wait(futures)
                
                # 確認全部發送完畢後才 Commit，確保不掉件
                consumer.commit()
                
        except KeyboardInterrupt:
            logger.info("收到中斷，關閉 consumer")
        finally:
            consumer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()