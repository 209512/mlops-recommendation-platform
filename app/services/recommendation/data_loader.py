import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from app.infrastructure.redis import get_redis_client
from app.services.recommendation.config import ALSConfig
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)

# 타입 별칭
MatrixBundle = dict[str, np.ndarray | csr_matrix | list[int] | list[str]]
MatrixBundleExtended = dict[str, np.ndarray | csr_matrix | list[int] | list[str] | datetime]


def _calculate_time_decay(days_passed: float, config: ALSConfig) -> float:
    """시간 감쇠 계산"""
    return float(0.5 ** (days_passed / config.half_life_days))


class ALSDataLoader:
    """ALS 모델을 위한 데이터 로더"""

    def __init__(
        self,
        lecture_repo: LectureRepository,
        user_repo: UserRepository,
        bookmark_repo: BookmarkRepository,
        search_log_repo: SearchLogRepository,
        user_pref_repo: UserPreferenceRepository,
        config: ALSConfig,
    ):
        self.lecture_repo = lecture_repo
        self.user_repo = user_repo
        self.bookmark_repo = bookmark_repo
        self.search_log_repo = search_log_repo
        self.user_pref_repo = user_pref_repo
        self.config = config
        self.redis_client = get_redis_client()
        self.lock_key = f"als_training_lock_{self.config.model_version}"

    async def load_training_data(
        self, cutoff_days: int | None = None
    ) -> MatrixBundleExtended | None:
        """
        학습용 데이터 로드

        Args:
            cutoff_days: 데이터 컷오프 기간 (None이면 config 기본값 사용)

        Returns:
            매트릭스 번들 또는 None
        """
        if cutoff_days is None:
            cutoff_days = self.config.search_log_days_limit

        try:
            return await _build_user_item_matrix(
                self.lecture_repo,
                self.user_repo,
                self.bookmark_repo,
                self.search_log_repo,
                self.user_pref_repo,
                cutoff_days,
                self.config,
            )
        except Exception as e:
            logger.error(f"[DATA_LOADER] Failed to load training data: {e}", exc_info=True)
            return None

    async def get_lecture_category_map(self) -> dict[int, list[str]]:
        """
        강의-카테고리 매핑 조회 (캐시 포함)

        Returns:
            {lecture_id: [category_names]}
        """
        try:
            # 캐시 확인 - 캐시 키를 동적으로 생성
            cache_key = f"lecture_category_map_{self.config.model_version}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return dict(json.loads(cached))

                # 데이터 조회
            # 빈 리스트로 모든 강의 카테고리 조회
            lecture_categories = await self.lecture_repo.get_lecture_categories([])

            # 매핑 생성
            category_map = defaultdict(list)
            for lecture_id, category_name in lecture_categories.items():
                for name in category_name:
                    category_map[lecture_id].append(name)

                    # 캐시 저장
            self.redis_client.setex(
                cache_key,
                self.config.lecture_category_map_timeout,
                json.dumps(dict(category_map)),
            )

            return dict(category_map)

        except Exception as e:
            logger.error(f"[DATA_LOADER] Failed to get lecture category map: {e}", exc_info=True)
            return {}

    async def get_user_item_stats(self, user_id: int, item_id: int) -> dict[str, Any]:
        """
        사용자-아이템 통계 조회

        Args:
            user_id: 사용자 ID
            item_id: 아이템 ID

        Returns:
            통계 정보 딕셔너리
        """
        try:
            # 기본 강의 정보
            lecture = await self.lecture_repo.get_lecture_by_id(item_id)
            if not lecture:
                return {}

            # 북마크 수
            bookmark_count = await self.bookmark_repo.get_lecture_bookmark_count(item_id)

            # 검색 횟수
            search_count = await self.search_log_repo.get_lecture_search_count(item_id, days=30)

            return {
                "id": lecture.id,
                "title": lecture.title,
                "bookmark_count": bookmark_count,
                "search_count": search_count,
                "rating": lecture.rating,
                "review_count": lecture.review_count,
                "difficulty": lecture.difficulty,
                "platform": lecture.platform,
            }

        except Exception as e:
            logger.error(f"[DATA_LOADER] Failed to get item stats: {e}", exc_info=True)
            return {}


async def _build_user_item_matrix(
    lecture_repo: LectureRepository,
    user_repo: UserRepository,
    bookmark_repo: BookmarkRepository,
    search_log_repo: SearchLogRepository,
    user_pref_repo: UserPreferenceRepository,
    cutoff_days: int,
    config: ALSConfig,
) -> MatrixBundleExtended | None:
    """
    사용자-아이템 상호작용 행렬 구축

    Args:
        lecture_repo: 강의 Repository
        user_repo: 사용자 Repository
        bookmark_repo: 북마크 Repository
        search_log_repo: 검색로그 Repository
        user_pref_repo: 사용자 선호도 Repository
        cutoff_days: 컷오프 기간
        config: ALS 설정

    Returns:
        확장된 매트릭스 번들 또는 None (상호작용이 없는 경우)
    """
    logger.info(f"[DATA_LOADER] Building user-item matrix with {cutoff_days} days cutoff")
    current_time = datetime.now()

    # 1. 사용자 및 아이템 ID 매핑
    users = await user_repo.get_active_users(days=cutoff_days)
    lectures = await lecture_repo.get_active_lectures()

    # SQLAlchemy 모델의 ID 속성을 직접 사용하여 매핑
    user_id_map = {int(user.id): idx for idx, user in enumerate(users)}
    lecture_id_map = {int(lecture.id): idx for idx, lecture in enumerate(lectures)}

    # 2. 상호작용 데이터 수집
    interactions = []

    # 북마크 상호작용
    bookmarks = await bookmark_repo.get_recent_bookmarks(days=cutoff_days)
    for bookmark in bookmarks:
        bookmark_user_id = int(bookmark.user_id)
        bookmark_lecture_id = int(bookmark.lecture_id)

        if bookmark_user_id in user_id_map and bookmark_lecture_id in lecture_id_map:
            weight = config.interaction_weights["bookmark"]  # 수정된 속성명 사용
            days_old = (current_time - bookmark.created_at).days
            decayed_weight = weight * _calculate_time_decay(days_old, config)

            interactions.append(
                (user_id_map[bookmark_user_id], lecture_id_map[bookmark_lecture_id], decayed_weight)
            )

            # 검색 상호작용 - 모든 사용자 검색 상호작용 조회
    search_interactions = await search_log_repo.get_all_search_interactions(days=cutoff_days)
    for user_id, lecture_id, weight in search_interactions:
        search_user_id = int(user_id)
        search_lecture_id = int(lecture_id)

        if search_user_id in user_id_map and search_lecture_id in lecture_id_map:
            interactions.append(
                (user_id_map[search_user_id], lecture_id_map[search_lecture_id], float(weight))
            )

            # 3. 희소 행렬 생성
    if not interactions:
        logger.warning("[DATA_LOADER] No interactions found")
        return None

    rows, cols, data = zip(*interactions, strict=True)
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(lectures)))

    # 4. 결과 반환
    return {
        "users": [int(user.id) for user in users],
        "lectures": [int(lecture.id) for lecture in lectures],
        "matrix": user_item_matrix,
        "last_trained_at": current_time,
    }
