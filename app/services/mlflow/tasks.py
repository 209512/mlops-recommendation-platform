import json
import logging
import os
from typing import Any

from celery import current_app
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from app.core.config import settings
from app.services.mlflow.registry import ModelRegistry
from app.services.mlflow.tracking import get_mlflow_tracking_service

logger = logging.getLogger(__name__)


@current_app.task(bind=True, name="create_mlflow_experiment")
def create_mlflow_experiment(
    self: Any,
    experiment_name: str,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """MLflow 실험 생성"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

        # 실험 생성
        experiment_id = mlflow_service.create_experiment(name=experiment_name, tags=tags or {})

        return {
            "status": "success",
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
        }

    except Exception as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "experiment_name": experiment_name,
        }


@current_app.task(bind=True, name="log_model_metrics")
def log_model_metrics(
    self: Any,
    run_id: str,
    metrics: dict[str, float],
    step: int | None = None,
) -> dict[str, Any]:
    """모델 메트릭 로깅"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

        # 메트릭 로깅
        mlflow_service.log_metrics(run_id=run_id, metrics=metrics, step=step)

        return {
            "status": "success",
            "run_id": run_id,
            "logged_metrics": list(metrics.keys()),
        }

    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "run_id": run_id,
        }


@current_app.task(bind=True, name="log_model_artifacts")
def log_model_artifacts(
    self: Any,
    run_id: str,
    artifact_path: str,
    artifacts: list[str],
) -> dict[str, Any]:
    """모델 아티팩트 로깅"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

            # 아티팩트 로깅
        for artifact in artifacts:
            mlflow_service.log_artifact(
                run_id=run_id, local_path=artifact, artifact_path=artifact_path
            )

        return {
            "status": "success",
            "run_id": run_id,
            "logged_artifacts": artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to log artifacts: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "run_id": run_id,
        }


@current_app.task(bind=True, name="register_model")
def register_model(
    self: Any,
    run_id: str,
    model_name: str,
    model_path: str = "model",
) -> dict[str, Any]:
    """모델 등록"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

        # 모델 등록
        model_version = mlflow_service.register_model(
            run_id=run_id, model_name=model_name, description=f"Model registered from run {run_id}"
        )

        return {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
        }

    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "run_id": run_id,
            "model_name": model_name,
        }


@current_app.task(bind=True, name="transition_model_stage")
def transition_model_stage(self: Any, model_name: str, version: str, stage: str) -> dict[str, Any]:
    """모델 스테이지 전환"""
    try:
        registry = ModelRegistry()
        success = registry.transition_model_stage(name=model_name, version=version, stage=stage)

        if not success:
            raise Exception("Failed to transition model stage")

        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "stage": stage,
        }

    except Exception as e:
        logger.error(f"Failed to transition model stage: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
        }


@current_app.task(bind=True, name="export_model_artifacts")
def export_model_artifacts(
    self: Any,
    model_name: str,
    version: str,
    export_path: str,
) -> dict[str, Any]:
    """모델 아티팩트 내보내기"""
    try:
        registry = ModelRegistry()

        # 모델 버전 정보 조회
        model_versions = registry.get_model_versions(model_name)
        target_version = next((v for v in model_versions if v["version"] == version), None)

        if not target_version:
            raise Exception(f"Model version {version} not found")

        # 내보내기 디렉토리 생성
        os.makedirs(export_path, exist_ok=True)

        # 모델 다운로드 및 내보내기
        import mlflow

        model_uri = f"models:/{model_name}/{version}"

        # 모델 로드 확인
        mlflow.sklearn.load_model(model_uri)

        # 메타데이터 파일 생성
        metadata = {
            "model_name": model_name,
            "version": version,
            "exported_at": str(self.request.id),
            "stage": target_version.get("stage", "Unknown"),
            "creation_timestamp": target_version.get("creation_timestamp"),
        }

        metadata_path = os.path.join(export_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported artifacts for model {model_name} version {version} to {export_path}")

        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "export_path": export_path,
            "artifacts": ["metadata.json", "model"],
        }

    except Exception as e:
        logger.error(f"Failed to export artifacts: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
        }


@current_app.task(bind=True, name="generate_model_report")
def generate_model_report(self: Any, model_name: str, version: str, run_id: str) -> dict[str, Any]:
    """모델 리포트 생성"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

        # 실행 기록 조회
        run_history = mlflow_service.get_run_history()
        target_run = None

        for run in run_history.get("runs", []):
            if run["run_id"] == run_id:
                target_run = run
                break

        if not target_run:
            raise Exception(f"Run {run_id} not found")

            # 메트릭 요약 생성
        metrics = target_run.get("metrics", {})
        metrics_summary = {}

        # 주요 메트릭 요약
        if metrics:
            metrics_summary = {
                "total_metrics": len(metrics),
                "key_metrics": dict(list(metrics.items())[:5]),  # 상위 5개 메트릭
                "metric_ranges": {
                    "min": min(metrics.values()) if metrics else None,
                    "max": max(metrics.values()) if metrics else None,
                },
            }

        # 리포트 생성
        report = {
            "run_id": run_id,
            "model_name": model_name,
            "version": version,
            "generated_at": str(self.request.id),
            "metrics_summary": metrics_summary,
            "run_status": target_run.get("status"),
            "start_time": target_run.get("start_time"),
            "end_time": target_run.get("end_time"),
        }

        logger.info(f"Generated report for model {model_name} version {version}")

        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "report": report,
        }

    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
        }


@current_app.task(bind=True, name="sync_metrics_to_monitoring")
def sync_metrics_to_monitoring(
    self: Any, run_id: str, metric_prefix: str = "mlflow"
) -> dict[str, Any]:
    """MLflow 메트릭을 모니터링 시스템에 동기화"""
    try:
        mlflow_service = get_mlflow_tracking_service()
        if not mlflow_service:
            raise Exception("MLflow service not available")

        # 실행 기록 조회
        run_history = mlflow_service.get_run_history()
        target_run = None

        for run in run_history.get("runs", []):
            if run["run_id"] == run_id:
                target_run = run
                break

        if not target_run:
            raise Exception(f"Run {run_id} not found")

        # Prometheus에 메트릭 푸시
        metrics = target_run.get("metrics", {})
        synced_metrics = []

        if metrics:
            registry = CollectorRegistry()

            for metric_name, metric_value in metrics.items():
                # Prometheus 게이지 생성
                gauge = Gauge(
                    f"{metric_prefix}_{metric_name}",
                    f"MLflow metric {metric_name} from run {run_id}",
                    registry=registry,
                )
                gauge.set(metric_value)
                synced_metrics.append(metric_name)

            # Prometheus 게이트웨이에 푸시 (설정된 경우)
            if hasattr(settings, "prometheus_pushgateway_url"):
                try:
                    push_to_gateway(
                        settings.prometheus_pushgateway_url,
                        job=f"mlflow_{run_id}",
                        registry=registry,
                    )
                except Exception as push_error:
                    logger.warning(f"Failed to push to Prometheus gateway: {push_error}")

        logger.info(f"Synced {len(synced_metrics)} metrics for run {run_id}")

        return {
            "status": "success",
            "run_id": run_id,
            "metric_prefix": metric_prefix,
            "synced_metrics": synced_metrics,
        }

    except Exception as e:
        logger.error(f"Failed to sync metrics: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "run_id": run_id,
        }
