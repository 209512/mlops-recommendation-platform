import logging
from typing import Any

import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram

import mlflow
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

    def __init__(self) -> None:
        self.tracking_uri: str = settings.mlflow_tracking_uri or "http://localhost:5000"
        self.experiment_name: str = settings.mlflow_experiment_name
        self.client: MlflowClient | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """MLflow 클라이언트 초기화"""
        try:
            # MLflow 추적 URI 설정
            mlflow.set_tracking_uri(self.tracking_uri)

            # 클라이언트 생성
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

            # 실험 생성 또는 가져오기
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")

        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            raise

    def start_run(
        self,
        run_name: str | None = None,
        experiment_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        MLflow 실행 시작

        Args:
            run_name: 실행 이름
            experiment_id: 실험 ID
            tags: 실행 태그

        Returns:
            실행 ID 또는 None
        """
        try:
            if experiment_id is None:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id if experiment else None

            run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id, tags=tags)

            mlflow_active_runs.inc()
            mlflow_experiment_counter.labels(status="started").inc()

            logger.info(f"Started MLflow run: {run.info.run_id}")
            return str(run.info.run_id)

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            mlflow_experiment_counter.labels(status="failed").inc()
            return None

    def end_run(self, run_id: str, status: str = "FINISHED") -> bool:
        """
        MLflow 실행 종료

        Args:
            run_id: 실행 ID
            status: 종료 상태

        Returns:
            성공 여부
        """
        try:
            if self.client is None:
                logger.error("MLflow client not initialized")
                return False

            self.client.set_terminated(run_id, status)
            mlflow_active_runs.dec()
            mlflow_experiment_counter.labels(status=status.lower()).inc()

            logger.info(f"Ended MLflow run: {run_id} with status: {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to end MLflow run {run_id}: {e}")
            return False

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> bool:
        """
        메트릭 기록

        Args:
            run_id: 실행 ID
            metrics: 메트릭 딕셔너리
            step: 단계

        Returns:
            성공 여부
        """
        try:
            with mlflow.start_run(run_id=run_id):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=step)

            logger.info(f"Logged {len(metrics)} metrics to run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log metrics to run {run_id}: {e}")
            return False

    def log_parameters(self, run_id: str, parameters: dict[str, Any]) -> bool:
        """
        파라미터 기록

        Args:
            run_id: 실행 ID
            parameters: 파라미터 딕셔너리

        Returns:
            성공 여부
        """
        try:
            with mlflow.start_run(run_id=run_id):
                for param_name, param_value in parameters.items():
                    mlflow.log_param(param_name, str(param_value))

            logger.info(f"Logged {len(parameters)} parameters to run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log parameters to run {run_id}: {e}")
            return False

    def log_model(
        self,
        run_id: str,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        registered_model_name: str | None = None,
    ) -> bool:
        """
        모델 기록

        Args:
            run_id: 실행 ID
            model: 모델 객체
            model_name: 모델 이름
            model_type: 모델 타입 (sklearn, pytorch 등)
            registered_model_name: 등록된 모델 이름

        Returns:
            성공 여부
        """
        try:
            with mlflow.start_run(run_id=run_id):
                if model_type == "sklearn":
                    mlflow.sklearn.log_model(
                        model, model_name, registered_model_name=registered_model_name
                    )
                elif model_type == "pytorch":
                    mlflow.pytorch.log_model(
                        model, model_name, registered_model_name=registered_model_name
                    )
                else:
                    mlflow.sklearn.log_model(model, model_name)

            logger.info(f"Logged model {model_name} to run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log model to run {run_id}: {e}")
            return False

    def get_run_history(self, experiment_name: str | None = None) -> dict[str, Any]:
        """
        실행 기록 조회

        Args:
            experiment_name: 실험 이름

        Returns:
            실행 기록
        """
        try:
            if experiment_name is None:
                experiment_name = self.experiment_name

            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return {"runs": []}

            if self.client is None:
                logger.error("MLflow client not initialized")
                return {"runs": []}

            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])

            run_history = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                }
                run_history.append(run_info)

            return {"runs": run_history}

        except Exception as e:
            logger.error(f"Failed to get run history: {e}")
            return {"runs": []}

    def get_best_model(
        self,
        experiment_name: str | None = None,
        metric_name: str = "accuracy",
        ascending: bool = False,
    ) -> str | None:
        """
        최고 성능 모델 조회

        Args:
            experiment_name: 실험 이름
            metric_name: 메트릭 이름
            ascending: 정렬 방향

        Returns:
            모델 URI 또는 None
        """
        try:
            if experiment_name is None:
                experiment_name = self.experiment_name

            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None

            if self.client is None:
                logger.error("MLflow client not initialized")
                return None

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1,
            )

            if runs:
                best_run = runs[0]
                model_uri = f"runs:/{best_run.info.run_id}/model"
                logger.info(
                    f"Found best model: {model_uri} with {metric_name}: "
                    f"{best_run.data.metrics.get(metric_name)}"
                )
                return model_uri

            return None

        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None

    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.client is None:
                logger.warning("MLflow client not initialized, skipping cleanup")
                return

            # 활성 실행 종료
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                active_runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id], filter_string="status = 'RUNNING'"
                )

                for run in active_runs:
                    self.client.set_terminated(run.info.run_id, "FAILED")

            logger.info("Cleaned up MLflow resources")

        except Exception as e:
            logger.error(f"Failed to cleanup MLflow resources: {e}")


def get_mlflow_tracking_service() -> MLflowTrackingService:
    """MLflow 추적 서비스 인스턴스 생성 (lazy initialization)"""
    return MLflowTrackingService()
