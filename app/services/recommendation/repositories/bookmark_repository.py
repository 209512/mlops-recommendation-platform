import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bookmark import Bookmark

logger = logging.getLogger(__name__)


class BookmarkRepository:
    """북마크 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_lecture_bookmark_count(self, lecture_id: int) -> int:
        """강의 북마크 수 조회"""
        query = select(func.count(Bookmark.id)).where(Bookmark.lecture_id == lecture_id)
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_user_bookmarks(self, user_id: int, limit: int = 100) -> list[Bookmark]:
        """사용자 북마크 조회"""
        result = await self.db.execute(
            select(Bookmark)
            .options(selectinload(Bookmark.lecture))
            .where(Bookmark.user_id == user_id)
            .order_by(Bookmark.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_lecture_bookmarks(self, lecture_id: int) -> list[Bookmark]:
        """강의 북마크 조회"""
        result = await self.db.execute(select(Bookmark).where(Bookmark.lecture_id == lecture_id))
        return list(result.scalars().all())

    async def get_user_bookmarked_lectures(self, user_id: int) -> list[int]:
        """사용자가 북마크한 강의 ID 목록"""
        result = await self.db.execute(
            select(Bookmark.lecture_id).where(Bookmark.user_id == user_id)
        )
        return [row[0] for row in result.all()]

    async def get_bookmark_stats(self) -> dict[str, Any]:
        """북마크 통계 조회"""
        total_bookmarks = await self.db.execute(select(func.count(Bookmark.id)))
        unique_users = await self.db.execute(select(func.count(func.distinct(Bookmark.user_id))))
        unique_lectures = await self.db.execute(
            select(func.count(func.distinct(Bookmark.lecture_id)))
        )

        return {
            "total_bookmarks": total_bookmarks.scalar(),
            "unique_users": unique_users.scalar(),
            "unique_lectures": unique_lectures.scalar(),
        }

    async def get_recent_bookmarks(self, days: int = 30) -> list[Bookmark]:
        """최근 북마크 조회"""
        cutoff_date = datetime.now() - timedelta(days=days)
        result = await self.db.execute(
            select(Bookmark)
            .options(selectinload(Bookmark.lecture))
            .where(Bookmark.created_at >= cutoff_date)
            .order_by(Bookmark.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_user_interaction_matrix(
        self, user_ids: list[int], lecture_ids: list[int]
    ) -> list[tuple[int, int, float]]:
        """사용자-강의 상호작용 매트릭스 데이터 조회 (ALS 학습용)"""
        result = await self.db.execute(
            select(Bookmark.user_id, Bookmark.lecture_id, func.count(Bookmark.id).label("weight"))
            .where(and_(Bookmark.user_id.in_(user_ids), Bookmark.lecture_id.in_(lecture_ids)))
            .group_by(Bookmark.user_id, Bookmark.lecture_id)
        )
        return [(row[0], row[1], float(row[2])) for row in result.all()]
