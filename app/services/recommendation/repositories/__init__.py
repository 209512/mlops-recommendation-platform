from app.services.recommendation.repositories.bookmark_repository import BookmarkRepository
from app.services.recommendation.repositories.lecture_repository import LectureRepository
from app.services.recommendation.repositories.search_log_repository import SearchLogRepository
from app.services.recommendation.repositories.user_preference_repository import (
    UserPreferenceRepository,
)
from app.services.recommendation.repositories.user_repository import UserRepository

__all__ = [
    "LectureRepository",
    "UserRepository",
    "BookmarkRepository",
    "SearchLogRepository",
    "UserPreferenceRepository",
]
