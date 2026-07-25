import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

@dataclass
class AlertPayload:
    """長照智慧預警系統標準化告警數據模型 (Dataclass Contract)"""
    alert_id: str
    device_id: int
    event_type: str        # 'fall', 'bed_exit', 'agitation', 'chair_slip', 'wandering'
    detected_at: str
    camera_id: str
    yolo_score: float
    vlm_summary: str
    severity: str          # 'critical', 'high', 'medium', 'low'
    status: str = "UNREAD"

    def to_dict(self) -> dict:
        """轉換為標準字典格式，方便 JSON 序列化外發 Kafka / WebSockets"""
        return asdict(self)

def build_alert_payload(
    prefix: str,
    camera_id: str,
    event_type: str,
    yolo_score: float,
    vlm_summary: str,
    severity: str = "medium"
) -> dict:
    """工廠函式：快速安全生成對齊後端/前端標準規格的 Payload 字典"""
    try:
        numeric_id = int(''.join(filter(str.isdigit, camera_id)))
    except ValueError:
        numeric_id = 1

    alert = AlertPayload(
        alert_id=f"{prefix}_{camera_id}_{int(time.time())}",
        device_id=numeric_id,
        event_type=event_type,
        detected_at=datetime.now().isoformat(),
        camera_id=camera_id,
        yolo_score=round(float(yolo_score), 4),
        vlm_summary=vlm_summary,
        severity=severity,
        status="UNREAD"
    )
    return alert.to_dict()
