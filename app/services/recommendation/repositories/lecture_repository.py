import logging
from collections import defaultdict

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bookmark import Bookmark
from app.models.lecture import Category, Lecture, LectureCategory

logger = logging.getLogger(__name__)


class LectureRepository:
    """강의 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_lecture_categories(self, lecture_ids: list[int]) -> dict[int, list[str]]:
        """강의별 카테고리 목록 조회"""
        query = (
            select(Lecture.id, Category.name)
            .join(LectureCategory, Lecture.id == LectureCategory.lecture_id)
            .join(Category, LectureCategory.category_id == Category.id)
            .where(Lecture.id.in_(lecture_ids))
        )

        result = await self.db.execute(query)
        categories = defaultdict(list)

        for lecture_id, category_name in result:
            categories[lecture_id].append(category_name)

        return dict(categories)

    async def get_lectures_by_ids(self, lecture_ids: list[int]) -> list[Lecture]:
        """ID 목록으로 강의 조회"""
        result = await self.db.execute(
            select(Lecture)
            .options(selectinload(Lecture.lecture_categories))
            .where(Lecture.id.in_(lecture_ids))
        )
        return list(result.scalars().all())

    async def get_lectures_by_category(self, category_id: int, limit: int = 10) -> list[Lecture]:
        """카테고리별 강의 조회"""
        result = await self.db.execute(
            select(Lecture)
            .options(selectinload(Lecture.lecture_categories))
            .join(LectureCategory)
            .where(LectureCategory.category_id == category_id)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_popular_lectures(self, limit: int = 10) -> list[Lecture]:
        """인기 강의 조회 (북마크 수 기준)"""
        result = await self.db.execute(
            select(Lecture, func.count(Bookmark.id).label("bookmark_count"))
            .options(selectinload(Lecture.lecture_categories))  # 주요 관계로 변경
            .join(Bookmark, Lecture.id == Bookmark.lecture_id)
            .group_by(Lecture.id)
            .order_by(func.count(Bookmark.id).desc())
            .limit(limit)
        )
        return [row[0] for row in result.all()]

    async def get_recent_lectures(self, limit: int = 10) -> list[Lecture]:
        """최신 강의 조회"""
        result = await self.db.execute(
            select(Lecture)
            .options(selectinload(Lecture.lecture_categories))
            .order_by(Lecture.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def search_lectures(self, keyword: str, limit: int = 10) -> list[Lecture]:
        """강의 검색"""
        result = await self.db.execute(
            select(Lecture)
            .options(selectinload(Lecture.lecture_categories))
            .where(
                or_(Lecture.title.ilike(f"%{keyword}%"), Lecture.description.ilike(f"%{keyword}%"))
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_lecture_by_id(self, lecture_id: int) -> Lecture | None:
        """ID로 강의 조회"""
        query = select(Lecture).where(Lecture.id == lecture_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_active_lectures(self, limit: int = 10000) -> list[Lecture]:
        """활성 강의 조회 (ALS 학습용)"""
        query = select(Lecture).where(Lecture.is_active).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_all_lectures(self) -> list[Lecture]:
        """전체 강의 조회 (ALS 학습용)"""
        result = await self.db.execute(
            select(Lecture).options(selectinload(Lecture.lecture_categories))
        )
        return list(result.scalars().all())

    async def get_lecture_count(self) -> int:
        """전체 강의 수 조회"""
        result = await self.db.execute(select(func.count(Lecture.id)))
        return int(result.scalar() or 0)
