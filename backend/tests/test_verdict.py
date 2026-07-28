# 測 PATCH /events/{id}/verdict：值班人員判定真跌倒/誤報
from backend.sse import pool


def test_未登入_401(client, make_event):
    event = make_event()
    res = client.patch(f"/events/{event.event_id}/verdict", json={"verdict": "false_alarm"})
    assert res.status_code == 401


def test_判誤報_幫事件結案(client, auth_headers, make_event):
    event = make_event()  # 預設 status=pending

    res = client.patch(
        f"/events/{event.event_id}/verdict",
        json={"verdict": "false_alarm"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "resolved"      # 誤報不用派人，直接結案
    assert data["verdict"] == "false_alarm"


def test_判真跌倒_進入處理中(client, auth_headers, make_event):
    event = make_event()

    res = client.patch(
        f"/events/{event.event_id}/verdict",
        json={"verdict": "true_alarm"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "in_progress"
    assert data["verdict"] == "true_alarm"


def test_已判定過再判_409(client, auth_headers, make_event):
    # 造一筆已經判定過的事件
    event = make_event(status="in_progress", verdict="true_alarm")
    res = client.patch(
        f"/events/{event.event_id}/verdict",
        json={"verdict": "false_alarm"},
        headers=auth_headers,
    )
    assert res.status_code == 409


def test_事件不存在_404(client, auth_headers):
    res = client.patch(
        "/events/00000000-0000-0000-0000-000000000000/verdict",
        json={"verdict": "false_alarm"},
        headers=auth_headers,
    )
    assert res.status_code == 404
