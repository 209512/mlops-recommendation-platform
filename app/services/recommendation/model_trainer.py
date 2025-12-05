import logging
import pickle
import time
from typing import Any, TypedDict

import implicit  # type: ignore
from scipy.sparse import csr_matrix

from app.infrastructure.redis import get_redis_client
from app.services.recommendation.constants import (
    ALS_PARAMS,
    LOCK_TIMEOUT_SECONDS,
    MODEL_VERSION,
)

logger = logging.getLogger(__name__)


class ModelBundle(TypedDict):
    """ALS 모델 번들 타입 정의"""

    model: implicit.als.AlternatingLeastSquares
    user_to_idx: dict[int, int]
    lecture_to_idx: dict[int, int]
    users: list[int]
    lectures: list[int]
    matrix: csr_matrix
    last_trained_at: float
    training_time: float | None


class ALSTrainer:
    """ALS 모델 트레이너"""

    def __init__(self) -> None:
        self.redis_client = get_redis_client()
        self.model: implicit.als.AlternatingLeastSquares | None = None

    def train_model(self, matrix_bundle: dict[str, Any]) -> bool:
        """ALS 모델 학습"""
        try:
            # 분산 락 획득
            lock_key = f"als_training_lock_{MODEL_VERSION}"
            if not self._acquire_lock(lock_key, LOCK_TIMEOUT_SECONDS):
                logger.warning("[TRAINER] Training already in progress")
                return False

            # 데이터 추출
            matrix = matrix_bundle["matrix"]
            users = matrix_bundle["users"]
            lectures = matrix_bundle["lectures"]

            # ALS 모델 생성 및 학습
            self.model = implicit.als.AlternatingLeastSquares(
                factors=ALS_PARAMS.factors,
                regularization=ALS_PARAMS.regularization,
                iterations=ALS_PARAMS.iterations,
                calculate_training_loss=ALS_PARAMS.calculate_training_loss,
                use_gpu=False,  # CPU 버전으로 구현
            )

            start_time = time.time()
            self.model.fit(matrix, show_progress=False)
            training_time = time.time() - start_time

            logger.info(f"[TRAINER] Model training completed in {training_time:.2f} seconds")

            # 모델 번들 생성
            model_bundle: ModelBundle = {
                "model": self.model,
                "user_to_idx": {user: idx for idx, user in enumerate(users)},
                "lecture_to_idx": {lecture: idx for idx, lecture in enumerate(lectures)},
                "users": users,
                "lectures": lectures,
                "matrix": matrix,
                "last_trained_at": time.time(),
                "training_time": training_time,
            }

            # 저장
            return self._save_model(model_bundle)

        except Exception as e:
            logger.error(f"[TRAINER] Model training failed: {e}", exc_info=True)
            return False

    def load_model(self) -> ModelBundle | None:
        """저장된 모델 로드"""
        try:
            serialized = self.redis_client.get(f"als_model_{MODEL_VERSION}")
            if not serialized:
                return None

            model_bundle = pickle.loads(serialized)

            # TypedDict 타입 변환으로 MyPy 오류 해결
            return ModelBundle(
                model=model_bundle["model"],
                user_to_idx=model_bundle["user_to_idx"],
                lecture_to_idx=model_bundle["lecture_to_idx"],
                users=model_bundle["users"],
                lectures=model_bundle["lectures"],
                matrix=model_bundle["matrix"],
                last_trained_at=model_bundle["last_trained_at"],
                training_time=model_bundle.get("training_time"),
            )
        except Exception as e:
            logger.error(f"[TRAINER] Failed to load model: {e}", exc_info=True)
            return None

    def recommend(
        self, user_id: int, user_items: csr_matrix, n: int = 10
    ) -> tuple[list[int], list[float]]:
        """사용자에게 아이템 추천"""
        if not self.model:
            return [], []

        ids, scores = self.model.recommend(user_id, user_items, n=n)
        return ids, scores.tolist()

    def train_incremental_model(self, matrix_bundle: dict[str, Any]) -> bool:
        """증분 모델 학습"""
        try:
            if not self.model:
                return self.train_model(matrix_bundle)

            matrix = matrix_bundle["matrix"]
            self.model.fit(matrix, show_progress=False)
            return True
        except Exception as e:
            logger.error(f"[TRAINER] Incremental training failed: {e}")
            return False

    def get_model_info(self) -> dict[str, Any]:
        """모델 정보 조회"""
        try:
            model_bundle = self.load_model()
            if model_bundle:
                return {
                    "status": "loaded",
                    "last_trained_at": model_bundle.get("last_trained_at"),
                    "training_time": model_bundle.get("training_time"),
                    "user_count": len(model_bundle.get("users", [])),
                    "lecture_count": len(model_bundle.get("lectures", [])),
                }
            else:
                return {"status": "not_found"}
        except Exception as e:
            logger.error(f"[TRAINER] Failed to get model info: {e}")
            return {"status": "error", "message": str(e)}

    def _acquire_lock(self, lock_key: str, timeout: int) -> bool:
        """분산 락 획득"""
        for _attempt in range(3):
            acquired = self.redis_client.set(lock_key, "locked", nx=True, ex=timeout)
            if acquired:
                return True
            time.sleep(0.1)
        return False

    def _save_model(self, model_bundle: ModelBundle) -> bool:
        """모델 저장"""
        try:
            serialized = pickle.dumps(model_bundle)
            self.redis_client.set(f"als_model_{MODEL_VERSION}", serialized, ex=86400)  # 24시간 만료
            return True
        except Exception as e:
            logger.error(f"[TRAINER] Failed to save model: {e}", exc_info=True)
            return False
