import io
import json
import logging
import os
import time
from typing import Any, TypedDict, cast
from urllib.parse import urlparse

import implicit
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.pyfunc.model import PythonModel
from mypy_boto3_s3 import S3Client
from scipy.sparse import csr_matrix, load_npz, save_npz

from app.infrastructure.aws import get_s3_client
from app.infrastructure.redis import get_redis_client
from app.services.recommendation.config import ALSConfig

logger = logging.getLogger(__name__)

# 환경 설정
USE_S3_STORAGE = os.getenv("USE_S3_STORAGE", "true").lower() == "true"
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./models")


class ModelBundle(TypedDict):
    """ALS 모델 번들 타입 정의 - 메모리 최적화 버전"""

    model: implicit.als.AlternatingLeastSquares
    user_to_idx: dict[int, int]
    lecture_to_idx: dict[int, int]
    idx_to_lecture: dict[int, int]
    users: list[int]
    lectures: list[int]
    matrix_location: str
    # matrix 필드 제거 - 메모리 절약
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
    """ALS 모델 트레이너 - 메모리 최적화 버전"""

    def __init__(self, config: ALSConfig) -> None:
        self.config = config
        self.redis_client = get_redis_client()
        self.model: implicit.als.AlternatingLeastSquares | None = None
        self._s3_client: S3Client | None = None
        self.use_s3 = USE_S3_STORAGE
        self.storage_path = MODEL_STORAGE_PATH

        # S3 클라이언트는 필요할 때만 초기화 (중복 초기화 방지)
        self._s3_client = None

    @property
    def s3_client(self) -> S3Client | None:
        """S3 클라이언트 지연 초기화 - 중복 생성 방지"""
        if self.use_s3 and self._s3_client is None:
            try:
                self._s3_client = get_s3_client()
            except ImportError:
                logger.warning("AWS infrastructure not available, falling back to local storage")
                self.use_s3 = False
        return self._s3_client

    def train_model(self, matrix_bundle: dict[str, Any]) -> bool:
        """ALS 모델 학습 - 메모리 최적화"""
        try:
            # 분산 락 획득
            lock_key = f"als_training_lock_{self.config.model_version}"
            if not self._acquire_lock(lock_key, self.config.lock_timeout_seconds):
                logger.warning("[TRAINER] Training already in progress")
                return False

            # 데이터 추출
            matrix = matrix_bundle["matrix"]
            users = matrix_bundle["users"]
            lectures = matrix_bundle["lectures"]
            user_to_idx = matrix_bundle["user_to_idx"]
            lecture_to_idx = matrix_bundle["lecture_to_idx"]

            # ALS 모델 생성 및 학습
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.config.factors,
                regularization=self.config.regularization,
                iterations=self.config.iterations,
                calculate_training_loss=self.config.calculate_training_loss,
                use_gpu=False,  # CPU 버전으로 구현
            )

            start_time = time.time()
            self.model.fit(matrix, show_progress=False)
            training_time = time.time() - start_time

            logger.info(f"[TRAINER] Model training completed in {training_time:.2f} seconds")

            # matrix 저장 (별도로 저장)
            matrix_location = self._save_matrix(matrix, time.time())

            # 모델 번들 생성 - matrix 제외
            model_bundle: ModelBundle = {
                "model": self.model,
                "user_to_idx": user_to_idx,
                "lecture_to_idx": lecture_to_idx,
                "idx_to_lecture": {v: k for k, v in lecture_to_idx.items()},
                "users": users,
                "lectures": lectures,
                "matrix_location": matrix_location,
                # matrix 필드 제거로 메모리 절약
                "last_trained_at": time.time(),
                "training_time": training_time,
            }

            # 저장
            return self._save_model_to_mlflow(model_bundle)

        except Exception as e:
            logger.error(f"[TRAINER] Model training failed: {e}", exc_info=True)
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
                    lecture_data = new_interactions.T[
                        [i for i, lid in enumerate(new_lectures) if lid in existing_lecture_to_idx]
                    ]
                    existing_model.partial_fit_items(new_lecture_indices, lecture_data)

            training_time = time.time() - start_time

            # 기존 모델 업데이트
            self.model = existing_model

            # 메타데이터만 업데이트
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

    def load_model(self) -> ModelBundle | None:
        """저장된 모델 로드 - 메모리 효율적"""
        try:
            # Redis에서 메타데이터 로드
            metadata = self._load_metadata()
            if not metadata:
                return None

                # MLflow에서 모델 로드
            pyfunc_model = mlflow.pyfunc.load_model(
                model_uri=f"models:/als_recommendation_model/{metadata['model_version']}"
            )

            # matrix는 필요할 때만 로드 (지연 로딩)
            # matrix = self._load_matrix(metadata["matrix_location"])

            # ModelBundle 재구성 - matrix 제외
            model_bundle: ModelBundle = {
                "model": pyfunc_model.model,
                "user_to_idx": metadata["user_to_idx"],
                "lecture_to_idx": metadata["lecture_to_idx"],
                "idx_to_lecture": metadata["idx_to_lecture"],
                "users": metadata["users"],
                "lectures": metadata["lectures"],
                "matrix_location": metadata["matrix_location"],
                # matrix는 필요시에만 로드
                "last_trained_at": metadata["last_trained_at"],
                "training_time": metadata["training_time"],
            }

            self.model = pyfunc_model.model
            return model_bundle

        except Exception as e:
            logger.error(f"[TRAINER] Failed to load model: {e}", exc_info=True)
            return None

    def recommend(
        self, user_idx: int, user_items: csr_matrix, n: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """추천 생성"""
        if self.model is None:
            raise ValueError("Model not loaded")

        result = self.model.recommend(user_idx, user_items, n=n, filter_already_liked_items=True)
        return (np.array(result[0]), np.array(result[1]))

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
                    "storage_type": "s3" if self.use_s3 else "local",
                    "matrix_location": model_bundle.get("matrix_location"),
                }
            else:
                return {"status": "not_found"}
        except Exception as e:
            logger.error(f"[TRAINER] Failed to get model info: {e}")
            return {"status": "error", "message": str(e)}

    def _save_model_to_mlflow(self, model_bundle: ModelBundle) -> bool:
        """MLflow에 모델 저장 - 메모리 최적화"""
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
                        "factors": self.config.factors,
                        "iterations": self.config.iterations,
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

                # matrix는 이미 별도로 저장됨
                # 메타데이터를 Redis에 저장 (matrix 제외)
                metadata = {
                    "user_to_idx": model_bundle["user_to_idx"],
                    "lecture_to_idx": model_bundle["lecture_to_idx"],
                    "idx_to_lecture": model_bundle["idx_to_lecture"],
                    "users": model_bundle["users"],
                    "lectures": model_bundle["lectures"],
                    "matrix_location": model_bundle["matrix_location"],
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

    def _save_matrix(self, matrix: csr_matrix, timestamp: float) -> str:
        """matrix를 S3 또는 로컬에 저장"""
        if USE_S3_STORAGE:
            return self._save_matrix_to_s3(matrix, timestamp)
        else:
            return self._save_matrix_to_local(matrix, timestamp)

    def _save_matrix_to_s3(self, matrix: csr_matrix, timestamp: float) -> str:
        """matrix를 S3에 저장"""
        try:
            s3_client = get_s3_client()
            bucket_name = os.getenv("MODEL_BUCKET_NAME", "mlops-models")
            key = f"als/models/matrix_{int(timestamp)}.npz"

            buffer = io.BytesIO()
            save_npz(buffer, matrix)
            buffer.seek(0)

            s3_client.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())

            location = f"s3://{bucket_name}/{key}"
            logger.info(f"[TRAINER] Matrix saved to S3: {location}")
            return location

        except Exception as e:
            logger.error(f"[TRAINER] Failed to save matrix to S3: {e}")
            # S3 실패 시 로컬 폴백
            logger.info("[TRAINER] Falling back to local storage")
            return self._save_matrix_to_local(matrix, timestamp)

    def _save_matrix_to_local(self, matrix: csr_matrix, timestamp: float) -> str:
        """matrix를 로컬 파일 시스템에 저장"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)

            filename = f"matrix_{int(timestamp)}.npz"
            file_path = os.path.join(self.storage_path, filename)

            save_npz(file_path, matrix)

            logger.info(f"[TRAINER] Matrix saved locally: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"[TRAINER] Failed to save matrix locally: {e}")
            raise

    def _load_matrix(self, location: str) -> csr_matrix | None:
        """S3 또는 로컬에서 matrix 로드 - 필요시에만 호출"""
        try:
            if location.startswith("s3://"):
                return self._load_matrix_from_s3(location)
            else:
                return self._load_matrix_from_local(location)
        except Exception as e:
            logger.error(f"[TRAINER] Failed to load matrix from {location}: {e}")
            return None

    def _load_matrix_from_s3(self, s3_url: str) -> csr_matrix | None:
        """S3에서 matrix 로드"""
        try:
            s3_client = get_s3_client()
            bucket_name = os.getenv("MODEL_BUCKET_NAME", "mlops-models")

            # S3 URL에서 key 추출
            parsed_url = urlparse(s3_url)
            key = parsed_url.path.lstrip("/")

            # S3에서 파일 다운로드
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            file_bytes = response["Body"].read()

            # BytesIO를 통해 npz 로드
            file_obj = io.BytesIO(file_bytes)
            matrix = load_npz(file_obj)

            result: csr_matrix = cast(csr_matrix, matrix)
            logger.info(f"[TRAINER] Matrix loaded from S3: {s3_url}")
            return result

        except Exception as e:
            logger.error(f"[TRAINER] Failed to load matrix from S3: {e}")
            return None

    def _load_matrix_from_local(self, file_path: str) -> csr_matrix | None:
        """로컬에서 matrix 로드"""
        try:
            matrix = load_npz(file_path)
            result: csr_matrix = cast(csr_matrix, matrix)
            return result
        except Exception as e:
            logger.error(f"[TRAINER] Failed to load matrix from {file_path}: {e}")
            return None

    def _acquire_lock(self, lock_key: str, timeout: int) -> bool:
        """분산 락 획득"""
        try:
            if self.redis_client:
                result = self.redis_client.set(lock_key, "locked", ex=timeout, nx=True)
                return bool(result)
            return True  # Redis가 없으면 락 없이 진행
        except Exception as e:
            logger.error(f"[TRAINER] Failed to acquire lock: {e}")
            return False

    def _release_lock(self, lock_key: str) -> None:
        """분산 락 해제"""
        try:
            if self.redis_client:
                self.redis_client.delete(lock_key)
        except Exception as e:
            logger.error(f"[TRAINER] Failed to release lock: {e}")

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """메타데이터를 Redis에 저장"""
        try:
            metadata_json = json.dumps(metadata, default=str)
            self.redis_client.set(
                f"als_metadata_{self.config.model_version}", metadata_json, ex=86400
            )
        except Exception as e:
            logger.error(f"[TRAINER] Failed to save metadata: {e}")
            raise

    def _load_metadata(self) -> dict[str, Any] | None:
        """Redis에서 메타데이터 로드"""
        try:
            metadata_json = self.redis_client.get(f"als_metadata_{self.config.model_version}")
            if not metadata_json:
                return None
            result: dict[str, Any] = json.loads(metadata_json)
            return result
        except Exception as e:
            logger.error(f"[TRAINER] Failed to load metadata: {e}")
            return None
