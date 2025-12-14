import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User

logger = logging.getLogger(__name__)


class UserRepository:
    """사용자 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> User | None:
        """ID로 사용자 조회 (관련 데이터 포함)"""
        result = await self.db.execute(
            select(User)
            .options(
                selectinload(User.bookmarks),
            )
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_all_users(self) -> list[User]:
        """전체 사용자 조회 (ALS 학습용, 관련 데이터 포함)"""
        result = await self.db.execute(
            select(User).options(
                selectinload(User.bookmarks),
            )
        )
        return list(result.scalars().all())

    async def get_user_count(self) -> int:
        """전체 사용자 수 조회"""
        result = await self.db.execute(select(func.count(User.id)))
        return int(result.scalar() or 0)

    async def get_active_users(self, days: int = 30) -> list[User]:
        """활성 사용자 조회 (최근 로그인 기준, 관련 데이터 포함)"""
        cutoff_date = datetime.now() - timedelta(days=days)
        result = await self.db.execute(
            select(User)
            .options(
                selectinload(User.bookmarks),
            )
            .where(User.last_login >= cutoff_date)
        )
        return list(result.scalars().all())

    async def get_users_for_recommendation(self, user_ids: list[int]) -> list[User]:
        """추천 시스템용 사용자 일괄 조회"""
        result = await self.db.execute(
            select(User)
            .options(
                selectinload(User.bookmarks),
            )
            .where(User.id.in_(user_ids))
        )
        return list(result.scalars().all())
