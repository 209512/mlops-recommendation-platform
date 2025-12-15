import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any

import implicit
import structlog
from celery import Task, current_app
from prometheus_client import Counter, Gauge, Histogram
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.infrastructure.database import get_async_db
from app.infrastructure.redis import get_redis_client
from app.schemas import RecommendationResponse
from app.services.mlflow.tracking import get_mlflow_tracking_service
from app.services.recommendation.data_loader import ALSDataLoader
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)
from app.services.recommendation.service import RecommendationService

# 구조화된 로거 설정
logger = structlog.get_logger(__name__)

# Prometheus 메트릭
task_counter = Counter(
    "celery_tasks_total", "Total number of Celery tasks", ["task_name", "status"]
)

task_duration = Histogram("celery_task_duration_seconds", "Duration of Celery tasks", ["task_name"])

active_tasks = Gauge("celery_active_tasks", "Number of active Celery tasks")


class TaskStatus(str, Enum):
    """태스크 상태 열거형"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


class BaseTask(Task[Any, Any]):
    """기본 Celery 태스크"""

    # 재시도 설정
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 5, "countdown": 60}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    # 타임아웃 설정 (초)
    time_limit = 3600  # 1시간
    soft_time_limit = 3000  # 50분

    def __init__(self) -> None:
        super().__init__()
        self.start_time: float | None = None
        self.task_name = self.name

    def on_success(
        self, retval: Any, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """성공 시 구조화된 로깅 및 메트릭"""
        duration = time.time() - (self.start_time or 0.0)

        logger.info(
            "Task completed successfully",
            task_name=self.task_name,
            task_id=task_id,
            duration=duration,
            result=retval,
        )

        # 메트릭 기록
        task_counter.labels(task_name=self.task_name, status="success").inc()
        task_duration.labels(task_name=self.task_name).observe(duration)
        active_tasks.dec()

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """실패 시 상세 로깅 및 알림"""
        duration = time.time() - (self.start_time or 0.0)

        logger.error(
            "Task failed",
            task_name=self.task_name,
            task_id=task_id,
            duration=duration,
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=str(einfo),
        )

        # 메트릭 기록
        task_counter.labels(task_name=self.task_name, status="failure").inc()
        task_duration.labels(task_name=self.task_name).observe(duration)
        active_tasks.dec()

        # 슬랙/이메일 알림 추가 가능
        self._send_failure_notification(exc, task_id)

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """재시도 시 로깅"""
        logger.warning(
            "Task retrying",
            task_name=self.task_name,
            task_id=task_id,
            retry_count=self.request.retries,
            max_retries=self.retry_kwargs["max_retries"],
            error=str(exc),
        )

        task_counter.labels(task_name=self.task_name, status="retry").inc()

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """태스크 실행 래퍼"""
        self.start_time = time.time()
        active_tasks.inc()

        logger.info(
            "Task started",
            task_name=self.task_name,
            task_id=self.request.id,
            args_count=len(args),
            kwargs_count=len(kwargs),
        )

        return super().run(*args, **kwargs)

    def _send_failure_notification(self, exc: Exception, task_id: str) -> None:
        """실패 알림 전송 (확장 가능)"""
        if settings.is_production:
            # 프로덕션 환경에서만 알림 전송
            logger.warning(
                "Production task failure notification would be sent",
                task_id=task_id,
                error=str(exc),
            )


@current_app.task(bind=True, base=BaseTask, name="train_als_model")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def train_als_model(
    self: BaseTask,
    model_name: str = "als_recommendation_model",
    experiment_name: str = "recommendation_experiments",
) -> dict[str, Any]:
    """ALS 모델 학습 Celery 태스크 - 안정성 강화"""
    run_id = None
    mlflow_service = None

    try:
        # 데이터 로더 초기화 위한 비동기 컨텍스트
        async def _train_model() -> dict[str, Any]:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
            try:
                # 리포지토리 초기화
                data_loader = ALSDataLoader(
                    lecture_repo=LectureRepository(db),
                    user_repo=UserRepository(db),
                    bookmark_repo=BookmarkRepository(db),
                    search_log_repo=SearchLogRepository(db),
                    user_pref_repo=UserPreferenceRepository(db),
                )

                # 사용자-아이템 행렬 로드
                matrix_bundle = await data_loader.load_training_data()
                if not matrix_bundle:
                    raise ValueError("No training data available")

                return matrix_bundle
            finally:
                await db_gen.aclose()

        # 데이터 로드 실행
        matrix_bundle = asyncio.run(_train_model())
        user_item_matrix = matrix_bundle["matrix"]

        # ALS 모델 초기화 및 학습
        als_model = implicit.als.AlternatingLeastSquares(
            factors=getattr(settings, "als_factors", 100),
            regularization=getattr(settings, "als_regularization", 0.01),
            iterations=getattr(settings, "als_iterations", 15),
            calculate_training_loss=True,
            random_state=42,  # 재현성을 위한 랜덤 시드
        )

        logger.info("Starting ALS model training", factors=als_model.factors)
        start_time = time.time()

        als_model.fit(user_item_matrix)

        training_time = time.time() - start_time
        logger.info("ALS model training completed", duration=training_time)

        # MLflow 서비스 가져오기
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise RuntimeError("MLflow service not available")

        # 실험 생성
        experiment_id = mlflow_service.create_experiment(
            name=experiment_name,
            tags={
                "model_type": "als",
                "environment": settings.environment,
                "task_id": self.request.id or "",
            },
        )

        # 실행 시작
        run_id = mlflow_service.start_run(run_name=f"{model_name}_{datetime.now().isoformat()}")

        if not run_id:
            raise RuntimeError("Failed to start MLflow run")

        try:
            # 모델 로깅
            model_info = mlflow_service.log_model(
                run_id=run_id, model=als_model, model_name=model_name, model_type="sklearn"
            )

            # 메트릭 로깅
            mlflow_service.log_metrics(
                run_id=run_id,
                metrics={
                    "training_time": float(training_time),
                    "matrix_shape": float(user_item_matrix.shape[0]),
                    "matrix_sparsity": 1.0
                    - (
                        user_item_matrix.nnz
                        / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
                    ),
                },
            )

            # 실행 종료 - 성공 상태로
            mlflow_service.end_run(run_id=run_id, status="FINISHED")
            logger.info("MLflow run completed successfully", run_id=run_id)

            return {
                "status": "success",
                "model_name": model_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "model_info": model_info,
                "experiment": experiment_name,
                "training_time": training_time,
            }

        except Exception as e:
            # 모델 로깅 중 에러 발생 시 run을 FAILED 상태로 종료
            mlflow_service.end_run(run_id=run_id, status="FAILED")
            logger.error("MLflow model logging failed", run_id=run_id, error=str(e))
            raise

    except Exception as e:
        # 최상위 예외 처리
        if run_id and mlflow_service:
            try:
                mlflow_service.end_run(run_id=run_id, status="FAILED")
                logger.error("MLflow run failed due to exception", run_id=run_id, error=str(e))
            except Exception as end_error:
                logger.error("Failed to end MLflow run", run_id=run_id, error=str(end_error))

        logger.error("ALS model training failed", error=str(e), exc_info=True)

        # 재시도 가능한 예외인 경우
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1)) from e

        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
            "task_id": self.request.id,
        }


@current_app.task(bind=True, base=BaseTask, name="generate_recommendations")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def generate_recommendations(
    self: BaseTask,
    user_id: int,
    num_recommendations: int = 10,
) -> dict[str, Any]:
    """사용자 추천 생성 - 안정성 강화"""
    try:
        # 입력값 검증
        if user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        if num_recommendations <= 0 or num_recommendations > 100:
            raise ValueError(f"Invalid num_recommendations: {num_recommendations}")

        # 비동기 컨텍스트 매니저 사용
        async def _generate_recommendations() -> list[RecommendationResponse] | dict[str, Any]:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
            try:
                # 리포지토리 초기화
                recommendation_service = RecommendationService(
                    lecture_repo=LectureRepository(db),
                    user_repo=UserRepository(db),
                    bookmark_repo=BookmarkRepository(db),
                    search_log_repo=SearchLogRepository(db),
                    user_pref_repo=UserPreferenceRepository(db),
                )

                # 추천 생성
                recommendations = await recommendation_service.get_recommendations(
                    user_id=user_id, num_recommendations=num_recommendations
                )

                return recommendations
            finally:
                await db_gen.aclose()

        # 비동기 함수 실행
        recommendations = asyncio.run(_generate_recommendations())

        logger.info(
            "Recommendations generated",
            user_id=user_id,
            count=len(recommendations),
            task_id=self.request.id,
        )

        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations) if recommendations else 0,
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(
            "Failed to generate recommendations",
            user_id=user_id,
            error=str(e),
            task_id=self.request.id,
            exc_info=True,
        )

        # 재시도 가능한 예외인 경우
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=30 * (self.request.retries + 1)) from e

        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
            "task_id": self.request.id,
        }


@current_app.task(bind=True, base=BaseTask, name="update_user_preferences")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
)
def update_user_preferences(
    self: BaseTask,
    user_id: int,
    category_id: int,
    preference_score: float,
) -> dict[str, Any]:
    """사용자 선호도 업데이트 - 안정성 강화"""
    try:
        # 입력값 검증
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        if not isinstance(category_id, int) or category_id <= 0:
            raise ValueError(f"Invalid category_id: {category_id}")

        if not isinstance(preference_score, (int, float)) or not 0 <= preference_score <= 1:
            raise ValueError(f"Invalid preference_score: {preference_score}")

        # 비동기 처리를 위한 래퍼
        async def _update_preferences() -> bool:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
            try:
                user_pref_repo = UserPreferenceRepository(db)

                # 선호도 업데이트
                result = await user_pref_repo.update_category_preference(
                    user_id=user_id, category_id=category_id
                )

                # preference_score는 별도로 업데이트
                if result:
                    # preference_score 업데이트는 별도 로직이 필요하면 구현해야 함
                    # 현재는 category_preference 업데이트만 수행
                    pass

                return bool(result)
            finally:
                await db_gen.aclose()

        # 비동기 실행
        success: bool = asyncio.run(_update_preferences())

        if success:
            logger.info(
                "User preference updated successfully",
                user_id=user_id,
                category_id=category_id,
                preference_score=preference_score,
                task_id=self.request.id,
            )

            return {
                "status": "success",
                "user_id": user_id,
                "category_id": category_id,
                "preference_score": preference_score,
                "task_id": self.request.id,
            }
        else:
            raise ValueError("Failed to update user preference")

    except Exception as e:
        logger.error(
            "Failed to update user preference",
            user_id=user_id,
            category_id=category_id,
            error=str(e),
            task_id=self.request.id,
            exc_info=True,
        )

        # 재시도 가능한 예외인 경우
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=30 * (self.request.retries + 1)) from e

        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
            "category_id": category_id,
            "task_id": self.request.id,
        }


@current_app.task(bind=True, base=BaseTask, name="batch_recommendations")
def batch_recommendations(
    self: BaseTask,
    user_ids: list[int],
    num_recommendations: int = 10,
) -> dict[str, Any]:
    """배치 추천 생성 - 안정성 강화"""
    try:
        # 입력값 검증
        if not user_ids:
            raise ValueError("user_ids cannot be empty")

        if len(user_ids) > 1000:
            raise ValueError("Batch size too large (max 1000)")

        if num_recommendations <= 0 or num_recommendations > 100:
            raise ValueError(f"Invalid num_recommendations: {num_recommendations}")

        results: list[dict[str, Any]] = []
        successful_count = 0
        failed_count = 0

        # 배치 처리
        for user_id in user_ids:
            try:
                # 개별 추천 생성 (동기 호출)
                result = generate_recommendations.apply_async(args=(user_id, num_recommendations))

                if result.ready() and result.successful():
                    result_data = result.get()
                    if result_data and result_data.get("status") == "success":
                        successful_count += 1
                        results.append(result_data)
                    else:
                        failed_count += 1
                        results.append(
                            {"status": "error", "error": "Task failed", "user_id": user_id}
                        )
                else:
                    failed_count += 1

            except Exception as e:
                error_msg = f"Failed to generate recommendations for user {user_id}: {str(e)}"
                logger.error(error_msg, task_id=self.request.id)

                failed_result: dict[str, Any] = {
                    "status": "error",
                    "error": str(e),
                    "user_id": user_id,
                }
                results.append(failed_result)
                failed_count += 1

        logger.info(
            "Batch recommendations completed",
            total_users=len(user_ids),
            successful=successful_count,
            failed=failed_count,
            task_id=self.request.id,
        )

        return {
            "status": "success",
            "total_users": len(user_ids),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "results": results,
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(
            "Batch recommendations failed",
            error=str(e),
            total_users=len(user_ids) if "user_ids" in locals() else 0,
            task_id=self.request.id,
            exc_info=True,
        )

        # 재시도 가능한 예외인 경우
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1)) from e

        return {
            "status": "error",
            "error": str(e),
            "total_users": len(user_ids) if "user_ids" in locals() else 0,
            "task_id": self.request.id,
        }


@current_app.task(bind=True, base=BaseTask, name="cleanup_expired_cache")
def cleanup_expired_cache(
    self: BaseTask,
    pattern: str = "recommendations:*",
    max_keys: int = 1000,
) -> dict[str, Any]:
    """만료된 캐시 정리 - 안정성 강화"""
    try:
        # 입력값 검증
        if not pattern:
            raise ValueError("pattern cannot be empty")

        if max_keys <= 0 or max_keys > 10000:
            raise ValueError(f"Invalid max_keys: {max_keys}")

        # 비동기 처리를 위한 래퍼
        async def _cleanup_cache() -> int:
            redis_client = get_redis_client()

            # 패턴으로 키 찾기
            keys = redis_client.keys(pattern)

            if len(keys) > max_keys:
                keys = keys[:max_keys]
                logger.warning(f"Limiting cleanup to {max_keys} keys out of {len(keys)} found")

            # 키 삭제
            if keys:
                deleted_count = int(redis_client.delete(*keys))
                return deleted_count

            return 0

        # 비동기 실행
        deleted_count: int = asyncio.run(_cleanup_cache())

        logger.info(
            "Cache cleanup completed",
            pattern=pattern,
            deleted_count=deleted_count,
            task_id=self.request.id,
        )

        return {
            "status": "success",
            "pattern": pattern,
            "deleted_count": deleted_count,
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(
            "Failed to cleanup cache",
            pattern=pattern,
            error=str(e),
            task_id=self.request.id,
            exc_info=True,
        )

        # 재시도 가능한 예외인 경우
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1)) from e

        return {
            "status": "error",
            "error": str(e),
            "pattern": pattern,
            "task_id": self.request.id,
        }


# Celery 설정 강화
current_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30분
    task_soft_time_limit=25 * 60,  # 25분
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1시간
    task_acks_late=True,
    worker_disable_rate_limits=False,
)
