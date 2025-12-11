import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

logger = logging.getLogger(__name__)


class UserRepository:
    """사용자 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> User | None:
        """ID로 사용자 조회"""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_all_users(self) -> list[User]:
        """전체 사용자 조회 (ALS 학습용)"""
        result = await self.db.execute(select(User))
        return list(result.scalars().all())

    async def get_user_count(self) -> int:
        """전체 사용자 수 조회"""
        result = await self.db.execute(select(func.count(User.id)))
        return int(result.scalar() or 0)

    async def get_active_users(self, days: int = 30) -> list[User]:
        """활성 사용자 조회 (최근 로그인 기준)"""
        cutoff_date = datetime.now() - timedelta(days=days)
        query = select(User).where(User.last_login >= cutoff_date)
        result = await self.db.execute(query)
        return list(result.scalars().all())
