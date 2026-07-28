try:
    from backend.core.database import Base, engine, SessionLocal, get_db
except ModuleNotFoundError:
    from core.database import Base, engine, SessionLocal, get_db