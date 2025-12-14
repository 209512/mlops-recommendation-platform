from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """추천 요청 스키마"""

    limit: int = Field(10, gt=0, le=100)
    exclude_bookmarked: bool = True
    model_version: str | None = None


class RecommendationResponse(BaseModel):
    """개별 추천 응답 스키마"""

    lecture_id: int
    title: str
    instructor: str | None = None
    thumbnail_img_url: str | None = None
    platform: str
    difficulty: str | None = None
    original_price: int | None = None
    discount_price: int | None = None
    rating: float | None = None
    review_count: int | None = None
    score: float = 0.0
    categories: list[str] = []
    generated_at: datetime = Field(default_factory=datetime.now)


class RecommendationList(BaseModel):
    """추천 목록 응답 스키마"""

    user_id: int
    recommendations: list[RecommendationResponse]
    model_version: str
    generated_at: datetime
    total_count: int


class TrainingMetrics(BaseModel):
    """모델 학습 메트릭 스키마"""

    training_time: float = Field(..., gt=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    ndcg: float = Field(..., ge=0.0, le=1.0)
    memory_usage: float = Field(..., gt=0)
    user_count: int = Field(..., gt=0)
    lecture_count: int = Field(..., gt=0)
    interaction_count: int = Field(..., gt=0)


class ModelInfo(BaseModel):
    """모델 정보 응답 스키마"""

    model_version: str
    last_trained_at: datetime | None = None
    training_time: float | None = None
    metrics: TrainingMetrics | None = None
    is_loaded: bool
    user_count: int | None = None
    lecture_count: int | None = None


class TrainingRequest(BaseModel):
    """모델 학습 요청 스키마"""

    force_retrain: bool = False
    cutoff_days: int = Field(30, gt=0, le=365)
    hyperparameters: dict[str, Any] | None = None


class TrainingResponse(BaseModel):
    """모델 학습 응답 스키마"""

    status: str = Field(pattern="^(success|error|in_progress)$")
    message: str
    metrics: TrainingMetrics | None = None
    model_version: str | None = None
