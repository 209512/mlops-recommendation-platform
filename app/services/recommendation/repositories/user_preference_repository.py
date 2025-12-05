import logging
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_preference import UserPreferCategory

logger = logging.getLogger(__name__)


class UserPreferenceRepository:
    """사용자 선호도 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_preferred_categories(self, user_id: int, limit: int = 5) -> list[int]:
        """
        사용자 선호 카테고리 ID 조회

        Args:
            user_id: 사용자 ID
            limit: 조회 개수

        Returns:
            [category_id, ...]
        """
        query = (
            select(UserPreferCategory.category_id)
            .where(UserPreferCategory.user_id == user_id)
            .order_by(UserPreferCategory.preference_score.desc())
            .limit(limit)
        )

        result = await self.db.execute(query)
        return [row[0] for row in result]

    async def get_user_category_preferences(self, user_id: int) -> list[tuple[int, float]]:
        """
        사용자 카테고리 선호도 점수 조회

        Args:
            user_id: 사용자 ID

        Returns:
            [(category_id, preference_score), ...]
        """
        query = (
            select(UserPreferCategory.category_id, UserPreferCategory.preference_score)
            .where(UserPreferCategory.user_id == user_id)
            .order_by(UserPreferCategory.preference_score.desc())
        )

        result = await self.db.execute(query)
        return [(row[0], row[1]) for row in result]

    async def update_category_preference(
        self, user_id: int, category_id: int, score_increment: float = 0.1
    ) -> UserPreferCategory:
        """
        카테고리 선호도 업데이트

        Args:
            user_id: 사용자 ID
            category_id: 카테고리 ID
            score_increment: 증가할 점수

        Returns:
            업데이트된 선호도 객체
        """
        # 기존 선호도 조회
        query = select(UserPreferCategory).where(
            and_(
                UserPreferCategory.user_id == user_id, UserPreferCategory.category_id == category_id
            )
        )

        result = await self.db.execute(query)
        preference = result.scalar_one_or_none()

        if preference:
            # 기존 선호도가 있으면 점수 증가
            preference.preference_score += score_increment  # type: ignore
        else:
            # 없으면 새로 생성
            preference = UserPreferCategory(
                user_id=user_id, category_id=category_id, preference_score=score_increment
            )
            self.db.add(preference)

        await self.db.commit()
        await self.db.refresh(preference)

        return preference

    async def get_category_stats(self) -> list[dict[str, Any]]:
        """
        카테고리별 선호도 통계 조회

        Returns:
            [{"category_id": int, "user_count": int, "avg_score": float}, ...]
        """
        query = (
            select(
                UserPreferCategory.category_id,
                func.count(UserPreferCategory.user_id).label("user_count"),
                func.avg(UserPreferCategory.preference_score).label("avg_score"),
            )
            .group_by(UserPreferCategory.category_id)
            .order_by(func.count(UserPreferCategory.user_id).desc())
        )

        result = await self.db.execute(query)
        return [
            {"category_id": row[0], "user_count": row[1], "avg_score": float(row[2])}
            for row in result
        ]
