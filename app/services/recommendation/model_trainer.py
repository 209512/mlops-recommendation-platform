import json
import logging
import time
from typing import Any, TypedDict

import implicit
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.pyfunc.model import PythonModel
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
    idx_to_lecture: dict[int, int]
    users: list[int]
    lectures: list[int]
    matrix: csr_matrix
    last_trained_at: float
    training_time: float | None


class ALSModelWrapper(PythonModel):
    """ALS 모델을 위한 MLflow PythonModel 래퍼"""

    def __init__(
        self,
        model: implicit.als.AlternatingLeastSquares,
        user_to_idx: dict[int, int],
        lecture_to_idx: dict[int, int],
        idx_to_lecture: dict[int, int],
    ):
        self.model = model
        self.user_to_idx = user_to_idx
        self.lecture_to_idx = lecture_to_idx
        self.idx_to_lecture = idx_to_lecture

    def load_context(self, context: Any) -> None:
        """모델 로딩 시 호출되는 메서드"""
        pass

    def predict(
        self,
        context: Any,
        model_input: dict[str, Any] | list[int],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        추천 예측 수행

        Args:
            model_input: {'user_id': int, 'limit': int} 또는 [user_id, limit]

        Returns:
            {'lecture_ids': list[int], 'scores': list[float]}
        """
        try:
            # 입력 파싱
            if isinstance(model_input, dict):
                user_id = model_input.get("user_id")
                limit = model_input.get("limit", 10)
            elif isinstance(model_input, list) and len(model_input) >= 2:
                user_id, limit = model_input[0], model_input[1]
            else:
                raise ValueError("Invalid input format")

            # 사용자 ID를 인덱스로 변환
            if user_id is None:
                raise ValueError("user_id is required")
            user_idx = self.user_to_idx.get(int(user_id))
            if user_idx is None:
                logger.warning(f"User {user_id} not found in model")
                return {"lecture_ids": [], "scores": []}

            # ALS 추천 생성
            user_items: csr_matrix = csr_matrix(
                ([1], ([0], [user_idx])), shape=(1, len(self.lecture_to_idx))
            )

            recommended_indices, scores = self.model.recommend(
                user_idx, user_items, N=limit, filter_already_liked_items=True
            )

            # 인덱스를 실제 강의 ID로 변환
            lecture_ids = [self.idx_to_lecture[idx] for idx in recommended_indices]

            return {
                "lecture_ids": lecture_ids,
                "scores": scores.tolist() if hasattr(scores, "tolist") else list(scores),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"lecture_ids": [], "scores": []}


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
            user_to_idx = matrix_bundle["user_to_idx"]
            lecture_to_idx = matrix_bundle["lecture_to_idx"]

            # 역매핑 딕셔너리 생성
            idx_to_lecture = {v: k for k, v in lecture_to_idx.items()}

            # ALS 모델 학습
            start_time = time.time()
            self.model = implicit.als.AlternatingLeastSquares(
                factors=ALS_PARAMS.factors,
                iterations=ALS_PARAMS.iterations,
                regularization=ALS_PARAMS.regularization,
                random_state=42,
            )

            self.model.fit(matrix)
            training_time = time.time() - start_time

            # 모델 번들 생성
            model_bundle: ModelBundle = {
                "model": self.model,
                "user_to_idx": user_to_idx,
                "lecture_to_idx": lecture_to_idx,
                "idx_to_lecture": idx_to_lecture,
                "users": users,
                "lectures": lectures,
                "matrix": matrix,
                "last_trained_at": time.time(),
                "training_time": training_time,
            }

            # MLflow에 모델 저장
            success = self._save_model_to_mlflow(model_bundle)

            return success

        except Exception as e:
            logger.error(f"[TRAINER] Training failed: {e}", exc_info=True)
            return False
        finally:
            self._release_lock(lock_key)

    def train_incremental_model(
        self,
        new_interactions: csr_matrix,
        new_users: list[int] | None = None,
        new_lectures: list[int] | None = None,
    ) -> dict[str, Any]:
        """실제 증분 학습 구현"""
        try:
            # 기존 모델 로드
            existing_bundle = self.load_model()
            if not existing_bundle:
                raise ValueError("No existing model found")

            existing_model = existing_bundle["model"]
            existing_user_to_idx = existing_bundle["user_to_idx"]
            existing_lecture_to_idx = existing_bundle["lecture_to_idx"]

            start_time = time.time()

            # 새 사용자 증분 학습
            if new_users:
                new_user_indices = [
                    existing_user_to_idx.get(uid, -1)
                    for uid in new_users
                    if uid in existing_user_to_idx
                ]
                if new_user_indices:
                    # 새 사용자들의 상호작용 데이터 추출
                    user_data = new_interactions[
                        [i for i, uid in enumerate(new_users) if uid in existing_user_to_idx]
                    ]
                    existing_model.partial_fit_users(new_user_indices, user_data)

                    # 새 강의 증분 학습
            if new_lectures:
                new_lecture_indices = [
                    existing_lecture_to_idx.get(lid, -1)
                    for lid in new_lectures
                    if lid in existing_lecture_to_idx
                ]
                if new_lecture_indices:
                    # 새 강의들의 상호작용 데이터 (전치)
                    lecture_data = new_interactions.T[
                        [i for i, lid in enumerate(new_lectures) if lid in existing_lecture_to_idx]
                    ]
                    existing_model.partial_fit_items(new_lecture_indices, lecture_data)

            training_time = time.time() - start_time

            # 기존 모델 업데이트
            self.model = existing_model

            # 메타데이터만 업데이트 (매핑 정보 변경 시)
            updated_bundle = existing_bundle.copy()
            updated_bundle["last_trained_at"] = time.time()
            updated_bundle["training_time"] = training_time

            # MLflow에 새 버전으로 저장
            success = self._save_model_to_mlflow(updated_bundle)

            return {
                "status": "success" if success else "error",
                "message": "Incremental training completed" if success else "Failed to save model",
                "training_time": training_time,
            }

        except Exception as e:
            logger.error(f"[TRAINER] Incremental training failed: {e}")
            return {"status": "error", "message": str(e)}

    def recommend(
        self, user_idx: int, user_items: csr_matrix, limit: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """추천 생성"""
        if self.model is None:
            raise RuntimeError("Model not trained")

        recommended_indices, scores = self.model.recommend(
            user_idx, user_items, N=limit, filter_already_liked_items=True
        )

        return recommended_indices, scores

    def load_model(self) -> ModelBundle | None:
        """모델 로드"""
        try:
            # MLflow에서 최신 모델 로드
            model_uri = "models:/als_recommendation_model/latest"  # 직접 URI 사용

            # MLflow 모델 로드
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            pyfunc_model = loaded_model._model_impl.python_model

            # 메타데이터 로드
            metadata = self._load_metadata()
            if not metadata:
                logger.warning("[TRAINER] No metadata found")
                return None

            # 모델 번들 재구성
            model_bundle: ModelBundle = {
                "model": pyfunc_model.model,
                "user_to_idx": metadata["user_to_idx"],
                "lecture_to_idx": metadata["lecture_to_idx"],
                "idx_to_lecture": metadata["idx_to_lecture"],
                "users": metadata["users"],
                "lectures": metadata["lectures"],
                "matrix": metadata["matrix"],
                "last_trained_at": metadata["last_trained_at"],
                "training_time": metadata["training_time"],
            }

            self.model = pyfunc_model.model
            return model_bundle

        except Exception as e:
            logger.error(f"[TRAINER] Failed to load model: {e}", exc_info=True)
            return None

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

    def _release_lock(self, lock_key: str) -> None:
        """분산 락 해제"""
        self.redis_client.delete(lock_key)

    def _save_model_to_mlflow(self, model_bundle: ModelBundle) -> bool:
        """MLflow에 모델 저장"""
        try:
            # 모델 래퍼 생성
            wrapper = ALSModelWrapper(
                model=model_bundle["model"],
                user_to_idx=model_bundle["user_to_idx"],
                lecture_to_idx=model_bundle["lecture_to_idx"],
                idx_to_lecture=model_bundle["idx_to_lecture"],
            )

            # MLflow에 모델 로깅
            with mlflow.start_run():
                # 모델 로깅
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    registered_model_name="als_recommendation_model",
                )

                # 파라미터 로깅
                mlflow.log_params(
                    {
                        "factors": ALS_PARAMS.factors,
                        "iterations": ALS_PARAMS.iterations,
                        "user_count": len(model_bundle["users"]),
                        "lecture_count": len(model_bundle["lectures"]),
                    }
                )

                # 메트릭 로깅
                mlflow.log_metrics(
                    {
                        "training_time": model_bundle["training_time"] or 0,
                    }
                )

                # 메타데이터를 Redis에 저장
                metadata = {
                    "user_to_idx": model_bundle["user_to_idx"],
                    "lecture_to_idx": model_bundle["lecture_to_idx"],
                    "idx_to_lecture": model_bundle["idx_to_lecture"],
                    "users": model_bundle["users"],
                    "lectures": model_bundle["lectures"],
                    "matrix": model_bundle["matrix"],
                    "last_trained_at": model_bundle["last_trained_at"],
                    "training_time": model_bundle["training_time"],
                    "model_version": model_info.model_version,
                }
                self._save_metadata(metadata)

                logger.info(
                    f"[TRAINER] Model saved to MLflow with version {model_info.model_version}"
                )
                return True

        except Exception as e:
            logger.error(f"[TRAINER] Failed to save model to MLflow: {e}", exc_info=True)
            return False

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """메타데이터를 Redis에 저장"""

        metadata_json = json.dumps(metadata, default=str)
        self.redis_client.set(f"als_metadata_{MODEL_VERSION}", metadata_json, ex=86400)

    def _load_metadata(self) -> dict[str, Any] | None:
        """Redis에서 메타데이터 로드"""
        try:
            metadata_json = self.redis_client.get(f"als_metadata_{MODEL_VERSION}")
            if not metadata_json:
                return None
            result: dict[str, Any] = json.loads(metadata_json)
            return result
        except Exception as e:
            logger.error(f"[TRAINER] Failed to load metadata: {e}")
            return None
