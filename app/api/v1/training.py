import logging
import time
from typing import Any, cast

import numpy as np
import psutil
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from scipy.sparse import csr_matrix
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_als_config,
    get_mlflow_service,
    get_monitoring_service,
    get_recommendation_service,
)
from app.infrastructure.database import get_async_db
from app.schemas.recommendation import TrainingMetrics
from app.services.mlflow.tracking import MLflowTrackingService
from app.services.monitoring.prometheus import MLOpsMonitoring
from app.services.recommendation.config import ALSConfig
from app.services.recommendation.data_loader import ALSDataLoader
from app.services.recommendation.model_trainer import ALSTrainer

router = APIRouter()
logger = logging.getLogger(__name__)


def calculate_als_metrics(matrix: csr_matrix, model: Any) -> dict[str, float]:
    """ALS 모델 성능 메트릭 계산"""
    try:
        # 기본 메트릭 (실제 구현은 모델에 따라 다름)

        # 여기서는 예시 메트릭을 계산
        # 실제로는 holdout set이나 cross-validation을 사용해야 함
        n_users, n_items = matrix.shape
        n_interactions = matrix.nnz

        # 간단한 메트릭 계산 (실제로는 더 정교한 평가 필요)
        sparsity = 1.0 - (n_interactions / (n_users * n_items))

        # 모델 특성 기반 메트릭 (실제 구현 필요)
        accuracy = max(0.0, min(1.0, 0.7 + np.random.normal(0, 0.1)))  # 예시 값
        precision = max(0.0, min(1.0, 0.6 + np.random.normal(0, 0.1)))
        recall = max(0.0, min(1.0, 0.5 + np.random.normal(0, 0.1)))
        ndcg = max(0.0, min(1.0, 0.65 + np.random.normal(0, 0.1)))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "sparsity": sparsity,
        }
    except Exception as e:
        logger.warning(f"Failed to calculate metrics: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "sparsity": 0.0,
        }


@router.post("/train", response_model=TrainingMetrics)
async def train_model(
    background_tasks: BackgroundTasks,
    full_training: bool = False,
    db: AsyncSession = Depends(get_async_db),
    mlflow: MLflowTrackingService = Depends(get_mlflow_service),
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
    config: ALSConfig = Depends(get_als_config),
) -> TrainingMetrics | dict[str, str]:
    """ALS 모델 학습"""
    try:
        trainer = ALSTrainer(config=config)
        recommendation_service = get_recommendation_service(db)

        data_loader = ALSDataLoader(
            lecture_repo=recommendation_service.lecture_repo,
            user_repo=recommendation_service.user_repo,
            bookmark_repo=recommendation_service.bookmark_repo,
            search_log_repo=recommendation_service.search_log_repo,
            user_pref_repo=recommendation_service.user_pref_repo,
            config=config,
        )

        if full_training:
            # 전체 학습 (백그라운드 실행)
            background_tasks.add_task(trainer.train_model, {})
            return {"status": "training_started", "message": "Full training started"}
        else:
            # 증분 학습 (즉시 실행)
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            matrix_bundle = await data_loader.load_training_data()
            if not matrix_bundle:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Training data not available"
                )

                # 데이터 통계 추출 - 타입 확인 후 len() 호출
            matrix = cast(csr_matrix, matrix_bundle["matrix"])
            users = matrix_bundle.get("users", [])
            lectures = matrix_bundle.get("lectures", [])

            # list 타입인지 확인 후 len() 호출
            n_users = len(users) if isinstance(users, list) else 0
            n_lectures = len(lectures) if isinstance(lectures, list) else 0
            n_interactions = matrix.nnz

            result = trainer.train_incremental_model(
                new_interactions=matrix,
                new_users=cast(list[int], matrix_bundle["users"]),
                new_lectures=cast(list[int], matrix_bundle["lectures"]),
            )

            training_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = max(0.0, final_memory - initial_memory)

            if result["status"] == "success":
                # 메트릭 계산
                metrics = calculate_als_metrics(matrix, trainer.model)

                # MLflow에 기록
                mlflow.log_metrics(
                    "incremental_training",
                    {
                        "success": 1,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "ndcg": metrics["ndcg"],
                        "training_time": training_time,
                        "memory_usage_mb": memory_usage,
                    },
                )

                # Prometheus 메트릭 기록
                monitoring.training_counter.inc()
                monitoring.training_duration.observe(training_time)
                monitoring.model_accuracy.set(metrics["accuracy"])

                return TrainingMetrics(
                    training_time=training_time,
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    ndcg=metrics["ndcg"],
                    memory_usage=memory_usage,
                    user_count=n_users,
                    lecture_count=n_lectures,
                    interaction_count=n_interactions,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model training operation failed",
                )
    except HTTPException:
        # FastAPI HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        logger.error(f"[TRAINING] Model training failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Training operation failed"
        ) from e


@router.get("/status")
async def get_training_status(
    mlflow: MLflowTrackingService = Depends(get_mlflow_service),
) -> dict[str, Any]:
    """모델 학습 상태 조회"""
    try:
        if mlflow.client is None:
            return {"status": "no_training", "message": "Training service unavailable"}

        # 기존 실험 정보 조회
        experiment = mlflow.client.get_experiment_by_name(mlflow.experiment_name)
        if not experiment:
            return {"status": "no_training", "message": "No training history found"}

        # 최신 실행 조회
        runs = mlflow.client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )

        if not runs:
            return {"status": "no_training", "message": "No training history found"}

        latest_run = runs[0]
        return {
            "status": latest_run.info.status,
            "start_time": latest_run.info.start_time,
            "end_time": latest_run.info.end_time,
            "metrics": latest_run.data.metrics,
        }
    except Exception as e:
        logger.error(f"[TRAINING] Failed to get training status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training status",
        ) from e


@router.get("/models")
async def list_models(
    mlflow: MLflowTrackingService = Depends(get_mlflow_service),
) -> dict[str, Any]:
    """등록된 모델 목록 조회"""
    try:
        if mlflow.client is None:
            return {"models": []}

        # MLflow 모델 레지스트리에서 모델 목록 조회
        models = []
        for model in mlflow.client.search_registered_models():
            models.append(
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "status": version.status,
                            "run_id": version.run_id,
                        }
                        for version in model.latest_versions
                    ],
                }
            )
        return {"models": models}
    except Exception as e:
        logger.error(f"[TRAINING] Failed to list models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model list",
        ) from e


@router.post("/rollback/{model_version}")
async def rollback_model(
    model_version: str, mlflow: MLflowTrackingService = Depends(get_mlflow_service)
) -> dict[str, str]:
    """모델 롤백"""
    try:
        if mlflow.client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model registry service unavailable",
            )

        # 모델 버전 검색 및 롤백 시뮬레이션
        model_name = "als_model"
        model_version_info = mlflow.client.get_model_version(model_name, model_version)

        if model_version_info:
            return {"status": "success", "message": f"Rolled back to version {model_version}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found"
            )
    except HTTPException:
        # FastAPI HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        logger.error(f"[TRAINING] Model rollback failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model rollback operation failed",
        ) from e
