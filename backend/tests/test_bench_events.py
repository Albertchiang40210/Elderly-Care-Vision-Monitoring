# 後端 POST /events 處理時間基準。預設跳過，執行：
#   BENCH=1 PYTHONPATH=..:. uv run pytest -s tests/test_bench_events.py
# 只量 FastAPI + in-memory SQLite + TestClient，不含 Kafka、網路、S3、AI 推論，
# 因此「不是」端到端延遲，只能說明後端本身不是瓶頸。
import os
import statistics
import time

import pytest

pytestmark = pytest.mark.skipif(os.getenv("BENCH") != "1", reason="設 BENCH=1 才執行")

N = 300


def test_post_events_latency(client):
    body = {"device_id": 1, "event_type": "fall", "clip_path": "s3://clips/b.mp4",
            "detected_at": "2026-07-02T14:30:00"}
    headers = {"X-API-Key": "test-api-key"}
    samples = []
    for _ in range(N):
        t = time.perf_counter()
        assert client.post("/events", json=body, headers=headers).status_code == 201
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    print(f"\nPOST /events N={N}: median={statistics.median(samples):.1f}ms "
          f"p95={samples[int(N*0.95)]:.1f}ms max={samples[-1]:.1f}ms")
