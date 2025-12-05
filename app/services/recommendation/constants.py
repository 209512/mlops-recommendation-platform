import os
from dataclasses import dataclass
from typing import Final

"""ALS 추천 시스템 상수 정의 모듈"""

# 모델 버전 관리
MODEL_VERSION: Final[str] = os.environ.get("ALS_MODEL_VERSION", "v1")

# 시간 감쇠 설정
HALF_LIFE_DAYS: Final[float] = 7.0
TIME_DECAY_ENABLED: Final[bool] = True


# ALS 하이퍼파라미터
@dataclass(frozen=True)
class ALSHyperparameters:
    factors: int = 50
    regularization: float = 0.1
    iterations: int = 15
    calculate_training_loss: bool = False


ALS_PARAMS: Final[ALSHyperparameters] = ALSHyperparameters()

# 상호작용 가중치
USER_INTERACTION_WEIGHTS: Final[dict[str, float]] = {
    "bookmark": 3.0,
    "search": 1.0,
    "study_participation": 2.0,
}

# 아이템 피처 가중치
ITEM_FEATURE_WEIGHTS: Final[dict[str, float]] = {
    "category_match": 2.5,
    "review_rating": 1.5,
}

# 후처리 상수
SEARCH_LOG_DAYS_LIMIT: Final[int] = 30
REVIEW_RATING_MULTIPLIER: Final[float] = 0.15
CATEGORY_MATCH_BONUS: Final[float] = 0.1
ALS_SCORE_POWER_DECAY: Final[float] = 0.8
NORMALIZE_ALS_SCORE: Final[bool] = True

# 캐시 키
LECTURE_CATEGORY_MAP_CACHE_KEY: Final[str] = "lecture_category_map"
LECTURE_CATEGORY_MAP_TIMEOUT: Final[int] = 3600
MODEL_CACHE_KEY: Final[str] = "als_model"

# 추천 설정
DEFAULT_RECOMMENDATION_COUNT: Final[int] = 10

# 락 설정
LOCK_TIMEOUT_SECONDS: Final[int] = 300
