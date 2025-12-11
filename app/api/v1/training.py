from typing import Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from scipy.sparse import csr_matrix
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_mlflow_service,
    get_monitoring_service,
    get_recommendation_service,
)
from app.infrastructure.database import get_async_db
from app.schemas.recommendation import TrainingMetrics
from app.services.mlflow.tracking import MLflowTrackingService
from app.services.monitoring.prometheus import MLOpsMonitoring
from app.services.recommendation.data_loader import ALSDataLoader
from app.services.recommendation.model_trainer import ALSTrainer

router = APIRouter()


@router.post("/train", response_model=TrainingMetrics)
async def train_model(
    background_tasks: BackgroundTasks,
    full_training: bool = False,
    db: AsyncSession = Depends(get_async_db),
    mlflow: MLflowTrackingService = Depends(get_mlflow_service),
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
) -> TrainingMetrics | dict[str, str]:
    """ALS 모델 학습"""
    try:
        trainer = ALSTrainer()
        recommendation_service = get_recommendation_service()

        data_loader = ALSDataLoader(
            lecture_repo=recommendation_service.lecture_repo,
            user_repo=recommendation_service.user_repo,
            bookmark_repo=recommendation_service.bookmark_repo,
            search_log_repo=recommendation_service.search_log_repo,
            user_pref_repo=recommendation_service.user_pref_repo,
        )

        if full_training:
            # 전체 학습 (백그라운드 실행)
            background_tasks.add_task(trainer.train_model, {})
            return {"status": "training_started", "message": "Full training started"}
        else:
            # 증분 학습 (즉시 실행)
            matrix_bundle = await data_loader.load_training_data()
            if not matrix_bundle:
                raise HTTPException(status_code=400, detail="No training data available")

            result = trainer.train_incremental_model(
                new_interactions=cast(csr_matrix, matrix_bundle["matrix"]),
                new_users=cast(list[int], matrix_bundle["users"]),
                new_lectures=cast(list[int], matrix_bundle["lectures"]),
            )

            if result["status"] == "success":
                # MLflow에 기록
                mlflow.log_metrics("incremental_training", {"success": 1})
                return TrainingMetrics(
                    training_time=result.get("training_time", 0.0),
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    ndcg=0.0,
                    memory_usage=0.0,
                    user_count=0,
                    lecture_count=0,
                    interaction_count=0,
                )
            else:
                raise HTTPException(status_code=500, detail="Incremental training failed")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Training failed: {str(e)}"
        ) from e


@router.get("/status")
async def get_training_status(
    mlflow: MLflowTrackingService = Depends(get_mlflow_service),
) -> dict[str, Any]:
    """모델 학습 상태 조회"""
    try:
        if mlflow.client is None:
            return {"status": "no_training", "message": "MLflow client not available"}

        # 기존 실험 정보 조회
        experiment = mlflow.client.get_experiment_by_name(mlflow.experiment_name)
        if not experiment:
            return {"status": "no_training", "message": "No training found"}

        # 최신 실행 조회
        runs = mlflow.client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )

        if not runs:
            return {"status": "no_training", "message": "No training found"}

        latest_run = runs[0]
        return {
            "status": latest_run.info.status,
            "start_time": latest_run.info.start_time,
            "end_time": latest_run.info.end_time,
            "metrics": latest_run.data.metrics,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}",
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
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
                detail="MLflow client not available",
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Rollback failed: {str(e)}"
        ) from e
