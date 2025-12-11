from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.services.mlflow.registry import ModelRegistry
from app.services.mlflow.tracking import get_mlflow_tracking_service

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


# Experiment endpoints
@router.post("/experiments")
async def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
    """MLflow 실험 생성"""
    try:
        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        experiment_id = service.create_experiment(name=request.name, tags=request.tags or {})
        return {"status": "success", "experiment_id": experiment_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/runs")
async def start_run(request: RunCreateRequest) -> dict[str, Any]:
    """MLflow 실행 시작"""
    try:
        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        run_id = service.start_run(
            experiment_name=request.experiment_name, run_name=request.run_name
        )
        return {"status": "success", "run_id": run_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/runs/{run_id}/metrics")
async def log_metrics(run_id: str, request: MetricsLogRequest) -> dict[str, Any]:
    """메트릭 로깅"""
    try:
        if not run_id or not run_id.strip():
            raise HTTPException(status_code=400, detail="run_id is required")

        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        success = service.log_metrics(run_id=run_id, metrics=request.metrics, step=request.step)
        if success:
            return {"status": "success", "logged_metrics": list(request.metrics.keys())}
        else:
            raise HTTPException(status_code=500, detail="Failed to log metrics")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/runs/{run_id}/parameters")
async def log_parameters(run_id: str, request: ParametersLogRequest) -> dict[str, Any]:
    """파라미터 로깅"""
    try:
        if not run_id or not run_id.strip():
            raise HTTPException(status_code=400, detail="run_id is required")

        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        success = service.log_parameters(run_id=run_id, parameters=request.parameters)
        if success:
            return {"status": "success", "logged_parameters": list(request.parameters.keys())}
        else:
            raise HTTPException(status_code=500, detail="Failed to log parameters")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/runs/{run_id}/model")
async def log_model(
    run_id: str,
    model_file: UploadFile | None = File(None, description="Model file to upload"),
    model_uri: str = Form(None, description="Model URI or path"),
    model_name: str = Form(..., description="Model name"),
    model_type: str = Form(default="sklearn", description="Model type"),
) -> dict[str, Any]:
    """모델 로깅"""
    try:
        if not run_id or not run_id.strip():
            raise HTTPException(status_code=400, detail="run_id is required")

        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        # Determine model source
        model_source = None
        if model_file:
            # Save uploaded file temporarily
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                content = await model_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Load model from file based on type
                if model_type == "sklearn":
                    import pickle

                    with open(tmp_file_path, "rb") as f:
                        model_source = pickle.load(f)
                elif model_type == "implicit":
                    # For ALS models, load the saved model
                    model_source = tmp_file_path
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        elif model_uri:
            # Use model URI directly
            model_source = model_uri

        else:
            raise HTTPException(
                status_code=400, detail="Either model_file or model_uri must be provided"
            )

        # Log the model
        success = service.log_model(
            run_id=run_id, model=model_source, model_name=model_name, model_type=model_type
        )

        if success:
            return {
                "status": "success",
                "message": f"Model '{model_name}' logged successfully",
                "model_type": model_type,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log model")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/register")
async def register_model(request: ModelRegisterRequest) -> dict[str, Any]:
    """모델 등록"""
    try:
        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        model_version = service.register_model(run_id=request.run_id, model_name=request.model_name)
        return {"status": "success", "model_version": model_version}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/{model_name}/versions/{version}/stage")
async def transition_model_stage(
    model_name: str, version: str, request: StageTransitionRequest
) -> dict[str, Any]:
    """모델 스테이지 전환"""
    try:
        if not model_name or not model_name.strip():
            raise HTTPException(status_code=400, detail="model_name is required")
        if not version or not version.strip():
            raise HTTPException(status_code=400, detail="version is required")

        registry = ModelRegistry()
        success = registry.transition_model_stage(
            name=model_name, version=version, stage=request.stage
        )
        if success:
            return {"status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to transition model stage")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/runs")
async def get_run_history(experiment_name: str = Query(..., min_length=1)) -> dict[str, Any]:
    """실행 기록 조회"""
    try:
        service = get_mlflow_tracking_service()
        if not service:
            raise HTTPException(status_code=503, detail="MLflow service unavailable")

        history = service.get_run_history(experiment_name=experiment_name)
        return {"status": "success", "runs": history.get("runs", [])}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str) -> dict[str, Any]:
    """모델 버전 조회"""
    try:
        if not model_name or not model_name.strip():
            raise HTTPException(status_code=400, detail="model_name is required")

        registry = ModelRegistry()
        versions = registry.get_model_versions(model_name)
        return {"status": "success", "versions": versions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/{model_name}/production")
async def get_production_model(model_name: str) -> dict[str, Any]:
    """프로덕션 모델 조회"""
    try:
        if not model_name or not model_name.strip():
            raise HTTPException(status_code=400, detail="model_name is required")

        registry = ModelRegistry()
        model_uri = registry.get_production_model(model_name)
        return {"status": "success", "model_uri": model_uri}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
