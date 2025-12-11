import logging
from typing import Any, cast

import mlflow
from mlflow import MlflowClient, sklearn
from prometheus_client import Counter, Gauge, Histogram

from app.core.config import settings

logger = logging.getLogger(__name__)

# Prometheus 메트릭
mlflow_experiment_counter = Counter(
    "mlflow_tracking_experiments_total", "Total number of MLflow experiments", ["status"]
)

mlflow_run_duration = Histogram(
    "mlflow_run_duration_seconds", "Duration of MLflow runs", ["experiment_name"]
)

mlflow_active_runs = Gauge("mlflow_active_runs", "Number of active MLflow runs")


class MLflowTrackingService:
    """MLflow 추적 서비스 - 실험 관리 및 모델 버전 관리"""

    def __init__(self, client: MlflowClient | None = None) -> None:
        """
        MLflow 추적 서비스 초기화

        Args:
            client: 선택적 MLflow 클라이언트. 테스트 시 mock 객체 주입에 사용.
                   제공되지 않으면 실제 클라이언트를 생성합니다.
        """
        self.tracking_uri: str = settings.mlflow_tracking_uri or "http://localhost:5001"
        self.experiment_name: str = settings.mlflow_experiment_name
        self.client: MlflowClient | None = client

        if client is None:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """MLflow 클라이언트 초기화"""
        try:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            logger.info(f"MLflow client initialized with URI: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            self.client = None

    def _ensure_client(self) -> bool:
        """클라이언트가 초기화되었는지 확인"""
        if self.client is None:
            self._initialize_client()
        return self.client is not None

    def create_experiment(self, name: str, tags: dict[str, str] | None = None) -> str | None:
        """실험 생성"""
        if not self._ensure_client():
            return None

        assert self.client is not None  # mypy를 위한 assertion
        try:
            experiment = self.client.get_experiment_by_name(name)
            if experiment is None:
                experiment_id = self.client.create_experiment(name, tags=tags or {})
                if experiment_id is not None:
                    logger.info(f"Created experiment: {name} with ID: {experiment_id}")
                    return experiment_id
                else:
                    return None
            else:
                logger.info(f"Experiment {name} already exists")
                return cast(str, experiment.experiment_id)
        except Exception as e:
            logger.error(f"Failed to create experiment {name}: {e}")
            return None

    def start_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """실행 시작"""
        if not self._ensure_client():
            return None

        assert self.client is not None  # mypy를 위한 assertion
        try:
            if experiment_name:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(experiment_name, tags=tags or {})
                    if experiment_id is None:
                        return None
                else:
                    experiment_id = experiment.experiment_id
            else:
                # 기본 실험 사용
                experiment_id = "0"

            run = self.client.create_run(experiment_id=experiment_id, run_name=run_name, tags=tags)
            logger.info(f"Started run: {run.info.run_id}")
            return cast(str, run.info.run_id)
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            return None

    def log_parameters(self, run_id: str, parameters: dict[str, Any]) -> bool:
        """파라미터 로깅"""
        if not self._ensure_client():
            return False

        assert self.client is not None
        try:
            for key, value in parameters.items():
                self.client.log_param(run_id, key, value)
            logger.info(f"Logged {len(parameters)} parameters to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            return False

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> bool:
        """메트릭 로깅"""
        if not self._ensure_client():
            return False

        assert self.client is not None
        try:
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, value, step=step)
            logger.info(f"Logged {len(metrics)} metrics to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False

    def log_model(
        self, run_id: str, model: Any, model_name: str, model_type: str = "sklearn"
    ) -> bool:
        """모델 로깅"""
        if not self._ensure_client():
            return False

        try:
            if model_type == "sklearn":
                # run_id를 사용하여 특정 실행에 모델 로깅
                with mlflow.start_run(run_id=run_id):
                    sklearn.log_model(model, model_name)
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                return False

            logger.info(f"Logged model {model_name} to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log model to run {run_id}: {e}")
            return False

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> bool:
        """아티팩트 로깅"""
        if not self._ensure_client():
            return False

        assert self.client is not None
        try:
            self.client.log_artifact(run_id, local_path, artifact_path)
            logger.info(f"Logged artifact {local_path} to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log artifact to run {run_id}: {e}")
            return False

    def end_run(self, run_id: str, status: str = "FINISHED") -> bool:
        """실행 종료"""
        if not self._ensure_client():
            return False

        assert self.client is not None
        try:
            self.client.set_terminated(run_id, status)
            logger.info(f"Ended run {run_id} with status {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to end run {run_id}: {e}")
            return False

    def get_run_history(self, experiment_name: str | None = None) -> dict[str, Any]:
        """실행 기록 조회"""
        if not self._ensure_client():
            return {"runs": []}

        assert self.client is not None
        try:
            if experiment_name:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    return {"runs": []}
                runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
            else:
                # 모든 실험 조회 (기본 실험 포함)
                runs = self.client.search_runs(experiment_ids=["0"])

            return {
                "runs": [
                    {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "metrics": run.data.metrics,
                    }
                    for run in runs
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get run history: {e}")
            return {"runs": []}

    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: str = "",
    ) -> str | None:
        """모델 등록"""
        if not self._ensure_client():
            return None

        assert self.client is not None
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=description,
            )
            return model_version.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def cleanup(self) -> None:
        """리소스 정리"""
        if not self._ensure_client():
            logger.warning("MLflow client not initialized, skipping cleanup")
            return

        assert self.client is not None
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                active_runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id], filter_string="status = 'RUNNING'"
                )

                for run in active_runs:
                    self.client.set_terminated(run.info.run_id, "FAILED")

            logger.info("Cleaned up MLflow resources")

        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")


def get_mlflow_tracking_service() -> MLflowTrackingService:
    """MLflow 추적 서비스 인스턴스 생성 (lazy initialization)"""
    return MLflowTrackingService()
