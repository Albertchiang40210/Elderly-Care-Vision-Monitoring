# test_register.py
# 測試 POST /register 路由的所有情況 (需要 admin 權限)

def _admin_headers(client):
    login = client.post("/login", data={"username": "boss", "password": "adminpass"})
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_register_new_user_returns_success_message(client):
    # admin 幫新員工開帳號
    headers = _admin_headers(client)
    response = client.post(
        "/register",
        json={"employee_id": "E888", "full_name": "新同事", "password": "password123"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "E888" in response.json()["message"]


def test_register_duplicate_username_returns_400(client):
    # alice 已經在 conftest.py 裡建好了，再次註冊同名帳號 → 應該回傳 400
    headers = _admin_headers(client)
    response = client.post(
        "/register",
        json={"employee_id": "alice", "full_name": "愛麗絲", "password": "password123"},
        headers=headers,
    )
    assert response.status_code == 400


def test_register_new_user_default_role_is_staff(client):
    headers = _admin_headers(client)
    client.post(
        "/register",
        json={"employee_id": "E999", "full_name": "新同仁", "password": "password123"},
        headers=headers,
    )
    login = client.post("/login", data={"username": "E999", "password": "password123"})
    assert login.status_code == 200
    assert "access_token" in login.json()


def test_register_without_password_returns_422(client):
    headers = _admin_headers(client)
    response = client.post(
        "/register",
        json={"employee_id": "E777", "full_name": "缺密碼"},
        headers=headers,
    )
    assert response.status_code == 422
