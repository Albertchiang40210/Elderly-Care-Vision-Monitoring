"""初始化正式資料庫（PostgreSQL）：建表 → 種 demo 資料 → 建初始帳號。

可重複執行：已存在的一律略過，不會報錯。
新環境（組員的機器、驗收主機、未來部署）第一次跑這支就能讓事件流程動起來。

用法：
    python -m backend.init_db
"""
from database import SessionLocal, Base, engine
from models import Company, Location, Device, Staff, User
from security import hash_password


def create_tables():
    """建立 models.py 定義的所有表（已存在的不動）。"""
    Base.metadata.create_all(bind=engine)
    print("表建立完成（已存在的不動）")


def seed_demo_data(db):
    """種入系統跑起來的最低內容物：公司 / 區域 / 裝置 / 照護員。

    沒有這些資料，POST /events 會因為查不到 device_id 而回 400，整個事件流程一步都跑不動。
    """
    if db.query(Company).filter_by(company_id=1).first() is None:
        db.add(Company(company_id=1, company_name="扶力憐示範安養院"))
        db.commit()
        print("已建立預設公司（id=1）")
    else:
        print("預設公司已存在，略過")

    locations = ["301 號病房", "302 號病房", "303 號病房", "交誼廳 A區", "走廊長廊", "門口區域", "護理站周邊"]
    for loc_name in locations:
        if db.query(Location).filter_by(location_name=loc_name).first() is None:
            db.add(Location(location_name=loc_name, company_id=1))
    db.commit()

    loc_ids = {l.location_name: l.location_id for l in db.query(Location).all()}
    target_devices = [
        {"device_id": 1, "device_name": "鏡頭 1 (301 號病房)", "location_name": "301 號病房"},
        {"device_id": 2, "device_name": "鏡頭 2 (302 號病房)", "location_name": "302 號病房"},
        {"device_id": 3, "device_name": "鏡頭 3 (303 號病房)", "location_name": "303 號病房"},
        {"device_id": 4, "device_name": "鏡頭 4 (交誼廳 A區)", "location_name": "交誼廳 A區"},
        {"device_id": 5, "device_name": "鏡頭 5 (走廊長廊)", "location_name": "走廊長廊"},
        {"device_id": 6, "device_name": "鏡頭 6 (門口區域)", "location_name": "門口區域"},
        {"device_id": 7, "device_name": "鏡頭 7 (護理站周邊)", "location_name": "護理站周邊"},
    ]
    for dev in target_devices:
        if db.query(Device).filter_by(device_id=dev["device_id"]).first() is None:
            db.add(Device(
                device_id=dev["device_id"],
                device_name=dev["device_name"],
                location_id=loc_ids.get(dev["location_name"]),
                status="active",
                company_id=1
            ))
            print(f"已建立示範裝置 id={dev['device_id']} ({dev['device_name']})")
    db.commit()

    if db.query(Staff).first() is None:
        db.add(Staff(staff_name="照護員A", company_id=1))
        db.add(Staff(staff_name="照護員B", company_id=1))
        db.commit()
        print("已建立照護員 2 名")
    else:
        print("照護員已存在，略過")


def seed_accounts(db):
    """建立可以登入中控站的初始帳號（A001 管理員 / E001 陳雅文，密碼皆 123456）。"""
    accounts = [
        {"name": "A001", "email": "a001@fulilian.com", "role": "admin"},
        {"name": "E001", "email": "e001@fulilian.com", "role": "staff"},
    ]
    for account in accounts:
        if db.query(User).filter(User.name == account["name"]).first():
            print(f"帳號 {account['name']} 已存在，略過建立")
        else:
            db.add(User(**account, password=hash_password("123456")))
            db.commit()
            print(f"帳號建立完成：{account['name']} / 123456（role: {account['role']}）")


if __name__ == "__main__":
    create_tables()

    db = SessionLocal()
    try:
        seed_demo_data(db)
        seed_accounts(db)
    finally:
        db.close()

    print("初始化完成")
