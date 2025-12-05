import logging
from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database import get_async_db
from app.services.mlflow import MLflowTrackingService
from app.services.monitoring import MLOpsMonitoring
from app.services.monitoring.prometheus import get_monitoring_service as _get_monitoring_service
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)
from app.services.recommendation.service import RecommendationService

logger = logging.getLogger(__name__)


def get_recommendation_service(db: AsyncSession = Depends(get_async_db)) -> RecommendationService:
    """추천 서비스 의존성 주입"""
    lecture_repo = LectureRepository(db)
    user_repo = UserRepository(db)
    bookmark_repo = BookmarkRepository(db)
    search_log_repo = SearchLogRepository(db)
    user_pref_repo = UserPreferenceRepository(db)

    return RecommendationService(
        lecture_repo=lecture_repo,
        user_repo=user_repo,
        bookmark_repo=bookmark_repo,
        search_log_repo=search_log_repo,
        user_pref_repo=user_pref_repo,
    )


def get_mlflow_service() -> Optional["MLflowTrackingService"]:
    """MLflow 서비스 의존성 주입"""
    try:
        from app.services.mlflow.tracking import get_mlflow_tracking_service

        return get_mlflow_tracking_service()
    except Exception as e:
        logger.error(f"Failed to initialize MLflow service: {e}")
        return None


def get_monitoring_service() -> Optional["MLOpsMonitoring"]:
    """모니터링 서비스 의존성 주입"""
    try:
        return _get_monitoring_service()
    except Exception as e:
        logger.error(f"Failed to initialize monitoring service: {e}")
        return None


async def get_current_user_id(user_id: int) -> int:
    """현재 사용자 ID 검증"""
    if not user_id:
        raise ValueError("User ID is required")
    return user_id
