import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_current_user_id,
    get_monitoring_service,
    get_recommendation_service,
)
from app.schemas.recommendation import (
    RecommendationList,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.monitoring.prometheus import MLOpsMonitoring
from app.services.recommendation.service import RecommendationService

router = APIRouter()


@router.post("/", response_model=RecommendationList)
async def get_recommendations(
    request: RecommendationRequest,
    user_id: int = Depends(get_current_user_id),
    service: RecommendationService = Depends(get_recommendation_service),
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
) -> RecommendationList:
    """사용자 맞춤 강의 추천"""
    try:
        recommendations = await service.get_recommendations(user_id=user_id, limit=request.limit)

        return RecommendationList(
            recommendations=recommendations,
            user_id=user_id,
            model_version="v1",
            generated_at=datetime.datetime.now(),
            total_count=len(recommendations),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}",
        ) from e


@router.get("/similar/{lecture_id}", response_model=list[RecommendationResponse])
async def get_similar_lectures(
    lecture_id: int,
    limit: int = 10,
    service: RecommendationService = Depends(get_recommendation_service),
) -> list[RecommendationResponse]:
    """유사 강의 추천"""
    try:
        similar_lectures = await service.get_similar_lectures(lecture_id=lecture_id, limit=limit)
        return similar_lectures
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get similar lectures: {str(e)}",
        ) from e


@router.post("/feedback")
async def record_feedback(
    user_id: int,
    lecture_id: int,
    feedback: bool,  # True: 좋아요, False: 싫어요
    service: RecommendationService = Depends(get_recommendation_service),
) -> dict[str, str]:
    """추천 피드백 기록"""
    try:
        await service.record_feedback(
            user_id=user_id, lecture_id=lecture_id, feedback={"rating": 1 if feedback else 0}
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        ) from e
