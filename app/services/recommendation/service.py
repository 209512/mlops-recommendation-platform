import asyncio
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from app.infrastructure.redis import get_redis_client
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

        # 프로세스 풀 실행기 (CPU 집약적 작업용)
        self.executor = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"))

    async def get_recommendations(
        self,
        user_id: int,
        num_recommendations: int = 10,
        exclude_bookmarked: bool = True,
        exclude_completed: bool = True,
    ) -> list[RecommendationResponse] | dict[str, Any]:
        """사용자 추천 목록 조회 (비동기 처리)"""
        try:
            # 캐시 확인
            cache_key = f"recommendations:{user_id}:{num_recommendations}"
            cached = get_redis_client().get(cache_key)

            if cached:
                recommendations_dict = json.loads(cached)
                recommendations = [RecommendationResponse(**rec) for rec in recommendations_dict]
                return recommendations

            # 비동기 추천 생성
            recommendations = await asyncio.to_thread(
                self._get_als_recommendations_sync, user_id, num_recommendations
            )

            # 캐시 저장 (5분)
            recommendations_dict = [rec.model_dump() for rec in recommendations]
            get_redis_client().setex(cache_key, 300, json.dumps(recommendations_dict))

            return recommendations

        except Exception as e:
            logger.error(f"[RECOMMENDATION] Failed to get recommendations: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def _get_als_recommendations(
        self, user_id: int, limit: int
    ) -> list[RecommendationResponse]:
        """ALS 기반 추천"""
        try:
            model_bundle = await asyncio.to_thread(self.trainer.load_model)

            if not model_bundle:
                return []

            # 사용자 ID를 행렬 인덱스로 변환
            user_idx = model_bundle["user_to_idx"].get(user_id)
            if user_idx is None:
                return []

            # matrix를 location에서 로드
            matrix_location = model_bundle.get("matrix_location")
            if not matrix_location:
                logger.error("[RECOMMENDATION] No matrix location in model bundle")
                return []

            matrix = await asyncio.to_thread(self.trainer._load_matrix, matrix_location)
            if matrix is None:
                logger.error("[RECOMMENDATION] Failed to load matrix")
                return []

            # 특정 사용자의 상호작용 데이터 추출
            user_items = matrix[user_idx]

            recommended_indices, scores = await asyncio.to_thread(
                self.trainer.recommend, user_idx, user_items, limit
            )

            # 인덱스를 실제 강의 ID로 변환
            idx_to_lecture = {v: k for k, v in model_bundle["lecture_to_idx"].items()}
            recommended_ids = [idx_to_lecture[idx] for idx in recommended_indices]

            # 강의 상세 정보 조회 (비동기)
            lectures = await self.lecture_repo.get_lectures_by_ids(recommended_ids)

            return self._convert_to_recommendations(
                lectures, scores.tolist() if hasattr(scores, "tolist") else list(scores)
            )

        except Exception as e:
            logger.error(f"[RECOMMENDATION] ALS recommendation failed: {e}", exc_info=True)
            return []

    def _get_als_recommendations_sync(
        self, user_id: int, limit: int
    ) -> list[RecommendationResponse]:
        """ALS 기반 추천"""
        try:
            # 동기 모델 로드
            model_bundle = self.trainer.load_model()

            if not model_bundle:
                return []

            # 사용자 ID를 행렬 인덱스로 변환
            user_idx = model_bundle["user_to_idx"].get(user_id)
            if user_idx is None:
                return []

            # matrix를 location에서 로드
            matrix_location = model_bundle.get("matrix_location")
            if not matrix_location:
                logger.error("[RECOMMENDATION] No matrix location in model bundle")
                return []

            # 동기 matrix 로드
            matrix = self.trainer._load_matrix(matrix_location)
            if matrix is None:
                logger.error("[RECOMMENDATION] Failed to load matrix")
                return []

            # 특정 사용자의 상호작용 데이터 추출
            user_items = matrix[user_idx]

            # 동기 ALS 추천 생성
            recommended_indices, scores = self.trainer.recommend(user_idx, user_items, limit)

            # 인덱스를 실제 강의 ID로 변환
            idx_to_lecture = {v: k for k, v in model_bundle["lecture_to_idx"].items()}
            recommended_ids = [idx_to_lecture[idx] for idx in recommended_indices]

            # 강의 상세 정보 조회는 비동기이므로 asyncio.run() 사용
            lectures = asyncio.run(self.lecture_repo.get_lectures_by_ids(recommended_ids))

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
            # 1. 피드백 데이터 검증
            if not feedback or "rating" not in feedback:
                logger.warning(f"Invalid feedback data for user {user_id}, lecture {lecture_id}")
                return False

            rating = feedback["rating"]
            if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
                logger.warning(f"Invalid rating value: {rating} for user {user_id}")
                return False

            # 2. 피드백 저장 (Repository 패턴 사용)
            await self.bookmark_repo.create_bookmark(user_id=user_id, lecture_id=lecture_id)

            # 3. 사용자 선호도 업데이트
            lecture = await self.lecture_repo.get_lecture_by_id(lecture_id)
            if lecture and lecture.categories:
                for category in lecture.categories:
                    await self.user_pref_repo.update_category_preference(
                        user_id=user_id, category_id=category.id
                    )

            # 4. 관련 캐시 무효화
            await self._clear_user_cache(user_id)

            logger.info(
                f"[FEEDBACK] Recorded feedback: "
                f"user={user_id}, lecture={lecture_id}, rating={rating}"
            )
            return True

        except Exception as e:
            logger.error(f"[FEEDBACK] Failed to record feedback: {e}", exc_info=True)
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
        """모델 학습"""
        try:
            # 데이터 로드
            matrix_bundle = await self.data_loader.load_training_data()
            if not matrix_bundle:
                return {"status": "error", "message": "Failed to load training data"}

            success = await asyncio.to_thread(self.trainer.train_model, matrix_bundle)

            # 학습 성공 시 캐시 클리어
            if success:
                await self._clear_all_recommendation_cache()

            return {"status": "success" if success else "error", "model_version": "latest"}

        except Exception as e:
            logger.error(f"[RECOMMENDATION] Model training failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def get_model_info(self) -> dict[str, Any]:
        """모델 정보 조회"""
        try:
            model_info = await asyncio.to_thread(self.trainer.get_model_info)
            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "error", "message": str(e)}

    async def cleanup(self) -> None:
        """리소스 정리"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    async def _clear_user_cache(self, user_id: int) -> None:
        """사용자 관련 캐시 클리어"""
        try:
            client = get_redis_client()
            if client:
                pattern = f"recommendations:{user_id}:*"
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to clear user cache: {e}")

    async def _clear_all_recommendation_cache(self) -> None:
        """전체 추천 캐시 클리어 (모델 재학습 후)"""
        try:
            client = get_redis_client()
            if client:
                pattern = "recommendations:*"
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} recommendation cache entries")
        except Exception as e:
            logger.warning(f"Failed to clear recommendation cache: {e}")
