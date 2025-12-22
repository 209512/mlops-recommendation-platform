import logging
from typing import Any, ClassVar

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ALSConfig(BaseSettings):
    """ALS 추천 시스템 설정"""

    model_config = SettingsConfigDict(
        env_prefix="ALS_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 상수 (변경 x)
    MODEL_TYPE: ClassVar[str] = "implicit"
    ALGORITHM_NAME: ClassVar[str] = "als"

    # 환경별 설정 (가끔 변경)
    model_version: str = Field(default="v1", description="모델 버전")
    factors: int = Field(default=50, ge=10, le=200, description="ALS 잠재 요인 수")
    regularization: float = Field(default=0.1, ge=0.001, le=1.0, description="정규화 강도")
    iterations: int = Field(default=15, ge=5, le=100, description="학습 반복 횟수")
    calculate_training_loss: bool = Field(default=False, description="학습 손실 계산 여부")

    # 시간 감쇠 설정
    half_life_days: float = Field(default=7.0, ge=1.0, le=30.0, description="시간 감쇠 반감기")
    time_decay_enabled: bool = Field(default=True, description="시간 감쇠 활성화")

    # 상호작용 가중치
    bookmark_weight: float = Field(default=3.0, ge=0.1, le=10.0, description="북마크 가중치")
    search_weight: float = Field(default=1.0, ge=0.1, le=10.0, description="검색 가중치")
    study_participation_weight: float = Field(
        default=2.0, ge=0.1, le=10.0, description="학습 참여 가중치"
    )

    # 아이템 피처 가중치
    category_match_weight: float = Field(
        default=2.5, ge=0.1, le=10.0, description="카테고리 매치 가중치"
    )
    review_rating_weight: float = Field(
        default=1.5, ge=0.1, le=10.0, description="리뷰 평점 가중치"
    )

    # 후처리 설정
    search_log_days_limit: int = Field(default=30, ge=1, le=90, description="검색 로그 보관 기간")
    review_rating_multiplier: float = Field(
        default=0.15, ge=0.01, le=1.0, description="리뷰 평점 배수"
    )
    category_match_bonus: float = Field(
        default=0.1, ge=0.01, le=1.0, description="카테고리 매치 보너스"
    )
    als_score_power_decay: float = Field(default=0.8, ge=0.1, le=1.0, description="ALS 점수 감쇠")
    normalize_als_score: bool = Field(default=True, description="ALS 점수 정규화")

    # 캐시 설정
    lecture_category_map_timeout: int = Field(
        default=3600, ge=60, le=86400, description="카테고리 맵 캐시 타임아웃"
    )

    # 추천 설정
    default_recommendation_count: int = Field(
        default=10, ge=1, le=100, description="기본 추천 개수"
    )

    # 락 설정
    lock_timeout_seconds: int = Field(default=300, ge=30, le=3600, description="락 타임아웃")

    # 실험 파라미터 (MLflow에서 관리)
    @property
    def experimental_params(self) -> dict[str, Any]:
        """MLflow 실험에서 관리할 파라미터"""
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "half_life_days": self.half_life_days,
            "bookmark_weight": self.bookmark_weight,
            "search_weight": self.search_weight,
            "study_participation_weight": self.study_participation_weight,
        }

    @property
    def interaction_weights(self) -> dict[str, float]:
        """상호작용 가중치 딕셔너리"""
        return {
            "bookmark": self.bookmark_weight,
            "search": self.search_weight,
            "study_participation": self.study_participation_weight,
        }

    @property
    def item_feature_weights(self) -> dict[str, float]:
        """아이템 피처 가중치 딕셔너리"""
        return {
            "category_match": self.category_match_weight,
            "review_rating": self.review_rating_weight,
        }

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, v: int) -> int:
        if v % 5 != 0:
            raise ValueError("factors는 5의 배수여야 함")
        return v

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_version은 비어있을 수 없음")
        return v.strip()

    def log_configuration(self) -> None:
        """설정 로깅"""
        logger.info(
            f"ALS Configuration loaded: model_version={self.model_version}, "
            f"factors={self.factors}, regularization={self.regularization}"
        )


# 전역 설정 인스턴스
als_config = ALSConfig()

# 설정 로딩 시 로깅
als_config.log_configuration()
