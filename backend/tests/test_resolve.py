# 測 PATCH /events/{id}/resolve：值班人員標記結案

def test_未登入_401(client, make_event):
    event = make_event(status="in_progress", verdict="true_alarm")
    res = client.patch(f"/events/{event.event_id}/resolve")
    assert res.status_code == 401


def test_處理中的事件可以結案(client, auth_headers, make_event):
    event = make_event(status="in_progress", verdict="true_alarm")

    res = client.patch(f"/events/{event.event_id}/resolve", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "resolved"
    assert data["verdict"] == "true_alarm"  # 判定結果不變，只改進度


def test_還沒判定就結案_409(client, auth_headers, make_event):
    event = make_event()  # status=pending
    res = client.patch(f"/events/{event.event_id}/resolve", headers=auth_headers)
    assert res.status_code == 409


def test_已結案再結案_409(client, auth_headers, make_event):
    event = make_event(status="resolved", verdict="false_alarm")
    res = client.patch(f"/events/{event.event_id}/resolve", headers=auth_headers)
    assert res.status_code == 409


def test_事件不存在_404(client, auth_headers):
    res = client.patch(
        "/events/00000000-0000-0000-0000-000000000000/resolve",
        headers=auth_headers,
    )
    assert res.status_code == 404
