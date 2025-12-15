import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import security
from app.infrastructure.database import get_async_db
from app.models.user import User
from app.services.mlflow import MLflowTrackingService
from app.services.mlflow.tracking import get_mlflow_tracking_service
from app.services.monitoring import MLOpsMonitoring
from app.services.monitoring.prometheus import get_monitoring_service as _get_monitoring_service
from app.services.recommendation.config import ALSConfig, als_config
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)
from app.services.recommendation.service import RecommendationService

logger = logging.getLogger(__name__)

# JWT Bearer 인증 스킴
security_scheme = HTTPBearer(auto_error=False)


def get_als_config() -> ALSConfig:
    """ALS 설정 의존성 주입"""
    try:
        return als_config
    except Exception as e:
        logger.error(f"Failed to load ALS configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ALS configuration unavailable",
        ) from e


def get_recommendation_service(
    db: AsyncSession = Depends(get_async_db), config: ALSConfig = Depends(get_als_config)
) -> RecommendationService:
    """추천 서비스 의존성 주입"""
    lecture_repo = LectureRepository(db)
    user_repo = UserRepository(db)
    bookmark_repo = BookmarkRepository(db)
    search_log_repo = SearchLogRepository(db)
    user_pref_repo = UserPreferenceRepository(db)

    return RecommendationService(
        lecture_repo, user_repo, bookmark_repo, search_log_repo, user_pref_repo, config
    )


def get_mlflow_service() -> Optional["MLflowTrackingService"]:
    """MLflow 서비스 의존성 주입"""
    try:
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


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> int:
    """현재 사용자 ID 검증 - JWT 토큰에서 추출"""

    # 인증 헤더가 없는 경우
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # JWT 토큰 검증
    try:
        token = credentials.credentials
        payload = security.verify_token(token)

        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 토큰에서 user_id 추출
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user_id",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return int(user_id)

    except HTTPException:
        # FastAPI HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user(
    user_id: int = Depends(get_current_user_id), db: AsyncSession = Depends(get_async_db)
) -> "User":
    """현재 사용자 객체 조회"""
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return user
