import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.search_log import LectureSearchLog

logger = logging.getLogger(__name__)


class SearchLogRepository:
    """검색 로그 관련 데이터 접근 객체"""

    def __init__(self, db: AsyncSession):
        self.db = db

    def _parse_clicked_lecture_ids(self, clicked_ids: str | None) -> list[int]:
        """
        JSON 형식의 clicked_lecture_ids를 파싱하는 헬퍼 함수

        Args:
            clicked_ids: JSON 문자열 또는 None

        Returns:
            강의 ID 리스트
        """
        try:
            return json.loads(clicked_ids) if clicked_ids else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse clicked_lecture_ids: {clicked_ids}")
            return []

    def _serialize_clicked_lecture_ids(self, lecture_ids: list[int] | None) -> str | None:
        """
        강의 ID 리스트를 JSON 문자열로 직렬화하는 헬퍼 함수

        Args:
            lecture_ids: 강의 ID 리스트 또는 None

        Returns:
            JSON 문자열 또는 None
        """
        return json.dumps(lecture_ids) if lecture_ids else None

    async def get_lecture_search_count(self, lecture_id: int, days: int = 30) -> int:
        """강의 검색 횟수 조회"""
        cutoff_date = datetime.now() - timedelta(days=days)
        query = select(func.count(LectureSearchLog.id)).where(
            LectureSearchLog.clicked_lecture_ids.like(f"%{lecture_id}%"),
            LectureSearchLog.created_at >= cutoff_date,
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_user_search_interactions(
        self, user_id: int, days: int = 30
    ) -> list[tuple[int, int, float]]:
        """
        사용자 검색 상호작용 데이터 조회

        Args:
            user_id: 사용자 ID
            days: 조회 기간

        Returns:
            ([(user_id, lecture_id, weight), ...]
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        query = select(
            LectureSearchLog.user_id,
            LectureSearchLog.clicked_lecture_ids,
            LectureSearchLog.search_duration,
        ).where(
            and_(
                LectureSearchLog.user_id == user_id,
                LectureSearchLog.created_at >= cutoff_date,
                LectureSearchLog.clicked_lecture_ids.isnot(None),
            )
        )

        result = await self.db.execute(query)
        interactions = []

        for row in result:
            clicked_ids = self._parse_clicked_lecture_ids(row.clicked_lecture_ids)
            weight = 1.0

            # 검색 시간이 길수록 가중치 증가
            if row.search_duration:
                weight = min(weight + (row.search_duration / 10.0), 3.0)

            for lecture_id in clicked_ids:
                interactions.append((row.user_id, lecture_id, weight))

        return interactions

    async def get_all_search_interactions(self, days: int = 30) -> list[tuple[int, int, float]]:
        """
        전체 사용자 검색 상호작용 데이터 조회

        Args:
            days: 조회 기간

        Returns:
            [(user_id, lecture_id, weight), ...]
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        query = select(
            LectureSearchLog.user_id,
            LectureSearchLog.clicked_lecture_ids,
            LectureSearchLog.search_duration,
        ).where(
            and_(
                LectureSearchLog.created_at >= cutoff_date,
                LectureSearchLog.clicked_lecture_ids.isnot(None),
            )
        )

        result = await self.db.execute(query)
        interactions = []

        for row in result:
            clicked_ids = self._parse_clicked_lecture_ids(row.clicked_lecture_ids)
            weight = 1.0

            if row.search_duration:
                weight = min(weight + (row.search_duration / 10.0), 3.0)

            for lecture_id in clicked_ids:
                interactions.append((row.user_id, lecture_id, weight))

        return interactions

    async def create_search_log(
        self,
        user_id: int,
        search_query: str,
        clicked_lecture_ids: list[int] | None = None,
        search_duration: float | None = None,
        result_count: int = 0,
    ) -> LectureSearchLog:
        """
        검색 로그 생성

        Args:
            user_id: 사용자 ID
            search_query: 검색어
            clicked_lecture_ids: 클릭한 강의 ID 리스트
            search_duration: 검색 소요 시간
            result_count: 검색 결과 수

        Returns:
            생성된 검색 로그
        """
        search_log = LectureSearchLog(
            user_id=user_id,
            search_query=search_query,
            clicked_lecture_ids=self._serialize_clicked_lecture_ids(clicked_lecture_ids),
            search_duration=search_duration,
            result_count=result_count,
        )

        self.db.add(search_log)
        await self.db.commit()
        await self.db.refresh(search_log)

        return search_log
