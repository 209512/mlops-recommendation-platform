import logging
import os
import pickle
import tempfile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.core.exception import ErrorResponse, MLflowServiceError, MLflowTrackingError
from app.services.mlflow.registry import ModelRegistry
from app.services.mlflow.tracking import get_mlflow_tracking_service

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Experiment name")
    tags: dict[str, str] | None = Field(default=None, description="Experiment tags")


class RunCreateRequest(BaseModel):
    experiment_name: str = Field(..., description="Experiment name")
    run_name: str | None = Field(default=None, description="Run name")


class MetricsLogRequest(BaseModel):
    metrics: dict[str, float] = Field(..., description="Metrics to log")
    step: int | None = Field(default=None, ge=0, description="Step number")


class ParametersLogRequest(BaseModel):
    parameters: dict[str, str] = Field(..., description="Parameters to log")


class ModelLogRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Model name")
    model_type: str = Field(default="sklearn", description="Model type")
    model_uri: str | None = Field(default=None, description="Model URI or path (optional)")


class ModelRegisterRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="Run ID")
    model_name: str = Field(..., min_length=1, description="Model name")


class StageTransitionRequest(BaseModel):
    stage: str = Field(..., min_length=1, description="Target stage")


def validate_required_string(value: str, field_name: str) -> None:
    """Validate required string field"""
    if not value or not value.strip():
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                message=f"{field_name} is required", error_code="VALIDATION_ERROR"
            ).model_dump(),
        )


def validate_mlflow_service() -> Any:
    """Validate MLflow service availability"""
    service = get_mlflow_tracking_service()
    if not service:
        raise MLflowServiceError("MLflow service unavailable").to_http_exception()
    return service


def handle_mlflow_exception(func_name: str, error: Exception) -> HTTPException:
    """MLflow 예외를 표준화된 HTTPException으로 변환"""
    error_msg = f"Failed to {func_name}"

    if isinstance(error, ValueError):
        logger.warning(f"{error_msg}: {str(error)}")
        return HTTPException(
            status_code=400,
            detail=ErrorResponse(
                message=error_msg,
                error_code="VALIDATION_ERROR",
                details={"original_error": str(error)},
            ).model_dump(),
        )
    elif isinstance(error, (MLflowServiceError, MLflowTrackingError)):
        logger.error(f"MLflow error in {func_name}: {str(error)}")
        return error.to_http_exception()
    elif "mlflow" in str(error).lower() or "tracking" in str(error).lower():
        logger.error(f"MLflow service error in {func_name}: {str(error)}")
        return MLflowServiceError(str(error)).to_http_exception()
    else:
        logger.error(f"Unexpected error in {func_name}: {str(error)}")
        return HTTPException(
            status_code=500,
            detail=ErrorResponse(
                message=error_msg,
                error_code="INTERNAL_ERROR",
                details={"original_error": str(error)},
            ).model_dump(),
        )


@router.post(
    "/experiments",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
    """MLflow 실험 생성"""
    try:
        service = validate_mlflow_service()

        experiment_id = service.create_experiment(name=request.name, tags=request.tags or {})
        logger.info(f"Created experiment: {experiment_id}")
        return {"status": "success", "experiment_id": experiment_id}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("create experiment", e) from e


@router.post(
    "/runs",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def start_run(request: RunCreateRequest) -> dict[str, Any]:
    """MLflow 실행 시작"""
    try:
        service = validate_mlflow_service()

        run_id = service.start_run(
            experiment_name=request.experiment_name, run_name=request.run_name
        )
        logger.info(f"Started run: {run_id}")
        return {"status": "success", "run_id": run_id}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("start run", e) from e


@router.post(
    "/runs/{run_id}/metrics",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def log_metrics(run_id: str, request: MetricsLogRequest) -> dict[str, Any]:
    """메트릭 로깅"""
    try:
        validate_required_string(run_id, "run_id")
        service = validate_mlflow_service()

        success = service.log_metrics(run_id=run_id, metrics=request.metrics, step=request.step)
        if success:
            logger.info(f"Logged metrics for run: {run_id}")
            return {"status": "success", "logged_metrics": list(request.metrics.keys())}
        else:
            raise MLflowTrackingError("Failed to log metrics", run_id=run_id)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("log metrics", e) from e


@router.post(
    "/runs/{run_id}/parameters",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def log_parameters(run_id: str, request: ParametersLogRequest) -> dict[str, Any]:
    """파라미터 로깅"""
    try:
        validate_required_string(run_id, "run_id")
        service = validate_mlflow_service()

        success = service.log_parameters(run_id=run_id, parameters=request.parameters)
        if success:
            logger.info(f"Logged parameters for run: {run_id}")
            return {"status": "success", "logged_parameters": list(request.parameters.keys())}
        else:
            raise MLflowTrackingError("Failed to log parameters", run_id=run_id)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("log parameters", e) from e


@router.post(
    "/runs/{run_id}/model",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def log_model(
    run_id: str,
    model_file: UploadFile | None = File(None, description="Model file to upload"),
    model_uri: str = Form(None, description="Model URI or path"),
    model_name: str = Form(..., description="Model name"),
    model_type: str = Form(default="sklearn", description="Model type"),
) -> dict[str, Any]:
    """모델 로깅"""
    try:
        validate_required_string(run_id, "run_id")
        service = validate_mlflow_service()

        model_source = None
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                content = await model_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                if model_type == "sklearn":
                    with open(tmp_file_path, "rb") as f:
                        model_source = pickle.load(f)
                elif model_type == "implicit":
                    model_source = tmp_file_path
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            finally:
                os.unlink(tmp_file_path)

        elif model_uri:
            model_source = model_uri

        else:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    message="Either model_file or model_uri must be provided",
                    error_code="VALIDATION_ERROR",
                ).model_dump(),
            )

        # Log the model
        success = service.log_model(
            run_id=run_id, model=model_source, model_name=model_name, model_type=model_type
        )

        if success:
            logger.info(f"Logged model '{model_name}' for run: {run_id}")
            return {
                "status": "success",
                "message": f"Model '{model_name}' logged successfully",
                "model_type": model_type,
            }
        else:
            raise MLflowTrackingError("Failed to log model", run_id=run_id)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("log model", e) from e


@router.post(
    "/models/register",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def register_model(request: ModelRegisterRequest) -> dict[str, Any]:
    """모델 등록"""
    try:
        service = validate_mlflow_service()

        model_version = service.register_model(run_id=request.run_id, model_name=request.model_name)
        logger.info(f"Registered model '{request.model_name}' version: {model_version}")
        return {"status": "success", "model_version": model_version}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("register model", e) from e


@router.post(
    "/models/{model_name}/versions/{version}/stage",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def transition_model_stage(
    model_name: str, version: str, request: StageTransitionRequest
) -> dict[str, Any]:
    """모델 스테이지 전환"""
    try:
        validate_required_string(model_name, "model_name")
        validate_required_string(version, "version")

        registry = ModelRegistry()
        success = registry.transition_model_stage(
            name=model_name, version=version, stage=request.stage
        )

        if success:
            logger.info(f"Transitioned model '{model_name}' v{version} to stage: {request.stage}")
            return {"status": "success"}
        else:
            raise MLflowTrackingError(
                f"Failed to transition model '{model_name}' v{version} to stage: {request.stage}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("transition model stage", e) from e


@router.get(
    "/runs",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_run_history(experiment_name: str = Query(..., min_length=1)) -> dict[str, Any]:
    """실행 기록 조회"""
    try:
        service = validate_mlflow_service()

        history = service.get_run_history(experiment_name=experiment_name)
        return {"status": "success", "runs": history.get("runs", [])}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("get run history", e) from e


@router.get(
    "/models/{model_name}/versions",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_model_versions(model_name: str) -> dict[str, Any]:
    """모델 버전 조회"""
    try:
        validate_required_string(model_name, "model_name")

        registry = ModelRegistry()
        versions = registry.get_model_versions(model_name)
        return {"status": "success", "versions": versions}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("get model versions", e) from e


@router.get(
    "/models/{model_name}/production",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_production_model(model_name: str) -> dict[str, Any]:
    """프로덕션 모델 조회"""
    try:
        validate_required_string(model_name, "model_name")

        registry = ModelRegistry()
        model_uri = registry.get_production_model(model_name)
        return {"status": "success", "model_uri": model_uri}

    except HTTPException:
        raise
    except Exception as e:
        raise handle_mlflow_exception("get production model", e) from e
