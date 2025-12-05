from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.base import Base

# 동기 SQLAlchemy 엔진 (Alembic 마이그레이션용)
engine = create_engine(
    settings.database_url or "", pool_pre_ping=True, pool_recycle=300, echo=settings.debug
)

# 비동기 SQLAlchemy 엔진
async_engine = create_async_engine(
    (settings.database_url or "").replace("postgresql://", "postgresql+asyncpg://"),
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug,
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """동기 데이터베이스 세션 의존성 주입"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """비동기 데이터베이스 세션 의존성 주입"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables() -> None:
    """테이블 생성"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
