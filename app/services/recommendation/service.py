import logging
from typing import Any

from app.schemas.recommendation import RecommendationResponse
from app.services.recommendation.data_loader import ALSDataLoader
from app.services.recommendation.model_trainer import ALSTrainer
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """추천 서비스"""

    def __init__(
        self,
        lecture_repo: LectureRepository,
        user_repo: UserRepository,
        bookmark_repo: BookmarkRepository,
        search_log_repo: SearchLogRepository,
        user_pref_repo: UserPreferenceRepository,
    ):
        self.lecture_repo = lecture_repo
        self.user_repo = user_repo
        self.bookmark_repo = bookmark_repo
        self.search_log_repo = search_log_repo
        self.user_pref_repo = user_pref_repo

        # 데이터 로더, 트레이너 초기화
        self.data_loader = ALSDataLoader(
            lecture_repo, user_repo, bookmark_repo, search_log_repo, user_pref_repo
        )
        self.trainer = ALSTrainer()

    async def get_recommendations(
        self, user_id: int, limit: int = 10
    ) -> list[RecommendationResponse]:
        """
        사용자 맞춤 강의 추천 - 3단계 폴백 로직

        Args:
            user_id: 사용자 ID
            limit: 추천 개수

        Returns:
            추천 강의 리스트
        """
        try:
            # 1. ALS 추천 시도
            recommendations = await self._get_als_recommendations(user_id, limit)

            # 2. 결과 부족 시 폴백
            if len(recommendations) < limit:
                fallback_recs = await self._get_category_fallback(
                    user_id, limit - len(recommendations)
                )
                recommendations.extend(fallback_recs)

            # 3. 최종 개수 조정
            return recommendations[:limit]

        except Exception as e:
            logger.error(
                f"[RECOMMENDATION] Failed to get recommendations for user {user_id}: {e}",
                exc_info=True,
            )
            return await self._get_popular_fallback(limit)

    async def _get_als_recommendations(
        self, user_id: int, limit: int
    ) -> list[RecommendationResponse]:
        """ALS 기반 추천"""
        try:
            # 모델 로드
            model_bundle = self.trainer.load_model()
            if not model_bundle:
                return []

            # 사용자 ID를 행렬 인덱스로 변환
            user_idx = model_bundle["user_to_idx"].get(user_id)
            if user_idx is None:
                return []

            # 특정 사용자의 상호작용 데이터 추출
            user_items = model_bundle["matrix"][user_idx]

            # ALS 추천 생성
            recommended_indices, scores = self.trainer.recommend(user_idx, user_items, limit)

            # 인덱스를 실제 강의 ID로 변환
            idx_to_lecture = {v: k for k, v in model_bundle["lecture_to_idx"].items()}
            recommended_ids = [idx_to_lecture[idx] for idx in recommended_indices]

            # 강의 상세 정보 조회
            lectures = await self.lecture_repo.get_lectures_by_ids(recommended_ids)

            return self._convert_to_recommendations(
                lectures, scores.tolist() if hasattr(scores, "tolist") else list(scores)
            )

        except Exception as e:
            logger.error(f"[RECOMMENDATION] ALS recommendation failed: {e}", exc_info=True)
            return []

    async def _get_category_fallback(
        self, user_id: int, needed_count: int
    ) -> list[RecommendationResponse]:
        """카테고리 기반 폴백 추천"""
        try:
            # 사용자 선호 카테고리 조회
            preferred_categories = await self.user_pref_repo.get_user_preferred_categories(
                user_id, limit=3
            )

            if not preferred_categories:
                # 선호 카테고리가 없으면 인기 강의로 폴백
                return await self._get_popular_fallback(needed_count)

            # 선호 카테고리의 강의 조회
            category_lectures = []
            for pref in preferred_categories:
                # pref 객체에서 category_id 추출
                category_id = pref.category_id if hasattr(pref, "category_id") else pref
                lectures = await self.lecture_repo.get_lectures_by_category(
                    category_id, limit=needed_count // len(preferred_categories) + 1
                )
                category_lectures.extend(lectures)

            return self._convert_to_recommendations(category_lectures[:needed_count])
        except Exception as e:
            logger.error(f"[RECOMMENDATION] Category fallback failed: {e}", exc_info=True)
            return await self._get_popular_fallback(needed_count)

    async def _get_popular_fallback(self, needed_count: int) -> list[RecommendationResponse]:
        """인기 강의 폴백"""
        try:
            popular_lectures = await self.lecture_repo.get_popular_lectures(limit=needed_count)
            return self._convert_to_recommendations(popular_lectures)
        except Exception as e:
            logger.error(f"[RECOMMENDATION] Popular fallback failed: {e}", exc_info=True)
            return []

    async def get_similar_lectures(
        self, lecture_id: int, limit: int = 10
    ) -> list[RecommendationResponse]:
        """유사 강의 추천"""
        try:
            lecture = await self.lecture_repo.get_lecture_by_id(lecture_id)
            if not lecture:
                return []

            # 카테고리 기반 유사 강의 조회
            similar_lectures = await self.lecture_repo.get_lectures_by_category(
                lecture.categories[0].id if lecture.categories else 1, limit=limit
            )
            return self._convert_to_recommendations(similar_lectures)
        except Exception as e:
            logger.error(f"Failed to get similar lectures: {e}")
            return []

    async def record_feedback(
        self, user_id: int, lecture_id: int, feedback: dict[str, Any]
    ) -> bool:
        """피드백 기록"""
        try:
            # 피드백 저장 로직 구현
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    def _convert_to_recommendations(
        self, lectures: list[Any], scores: list[float] | None = None
    ) -> list[RecommendationResponse]:
        """강의 객체를 추천 응답으로 변환"""
        recommendations = []

        for i, lecture in enumerate(lectures):
            score = scores[i] if scores and i < len(scores) else 0.0

            recommendations.append(
                RecommendationResponse(
                    lecture_id=lecture.id,
                    title=lecture.title,
                    instructor=lecture.instructor,
                    thumbnail_img_url=lecture.thumbnail_img_url,
                    platform=lecture.platform,
                    difficulty=lecture.difficulty,
                    original_price=lecture.original_price,
                    discount_price=lecture.discount_price,
                    rating=lecture.rating,
                    review_count=lecture.review_count,
                    score=score,
                    categories=[cat.name for cat in lecture.categories],
                )
            )

        return recommendations

    async def train_model(self, force_retrain: bool = False) -> dict[str, Any]:
        """
        모델 학습

        Args:
            force_retrain: 강제 재학습 여부

        Returns:
            학습 결과
        """
        try:
            # 데이터 로드
            matrix_bundle = await self.data_loader.load_training_data()
            if not matrix_bundle:
                return {"status": "error", "message": "Failed to load training data"}

            # 모델 학습
            success = self.trainer.train_model(matrix_bundle)

            return {"status": "success" if success else "error", "model_version": "latest"}

        except Exception as e:
            logger.error(f"[RECOMMENDATION] Model training failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def get_model_info(self) -> dict[str, Any]:
        """모델 정보 조회"""
        try:
            model_info = self.trainer.get_model_info()
            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "error", "message": str(e)}
