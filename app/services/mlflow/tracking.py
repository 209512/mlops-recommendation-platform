import logging
import pickle
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import mlflow
from mlflow import MlflowClient, sklearn
from mlflow.config import enable_async_logging
from prometheus_client import Counter, Gauge, Histogram, Info

from app.core.config import settings

logger = logging.getLogger(__name__)

mlflow_experiment_counter = Counter(
    "mlflow_tracking_experiments_total", "Total number of MLflow experiments", ["status"]
)

mlflow_run_duration = Histogram(
    "mlflow_run_duration_seconds",
    "Duration of MLflow runs",
    ["experiment_name"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf")],
)

mlflow_active_runs = Gauge("mlflow_active_runs", "Number of active MLflow runs")


# 상세 운영 메트릭
mlflow_operations_total = Counter(
    "mlflow_operations_total", "Total MLflow operations", ["operation", "status", "experiment_name"]
)

mlflow_operation_duration = Histogram(
    "mlflow_operation_duration_seconds",
    "Duration of MLflow operations by type",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")],
)

# 모델 관련 메트릭
mlflow_model_logging_total = Counter(
    "mlflow_model_logging_total",
    "Total model logging operations",
    ["model_type", "status", "experiment_name"],
)

mlflow_model_size_bytes = Histogram(
    "mlflow_model_size_bytes",
    "Size of logged models in bytes",
    ["model_type"],
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824, float("inf")],
)

# 파라미터 및 메트릭 메트릭
mlflow_parameters_total = Counter(
    "mlflow_parameters_logged_total", "Total number of parameters logged", ["experiment_name"]
)

mlflow_metrics_total = Counter(
    "mlflow_metrics_logged_total", "Total number of metrics logged", ["experiment_name"]
)

# 아티팩트 메트릭
mlflow_artifacts_total = Counter(
    "mlflow_artifacts_logged_total",
    "Total number of artifacts logged",
    ["artifact_type", "status", "experiment_name"],
)

mlflow_artifact_size_bytes = Histogram(
    "mlflow_artifact_size_bytes",
    "Size of logged artifacts in bytes",
    ["artifact_type"],
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824, float("inf")],
)

# 실험 관련 메트릭
mlflow_experiments_active = Gauge("mlflow_experiments_active", "Number of active experiments")

mlflow_runs_per_experiment = Histogram(
    "mlflow_runs_per_experiment",
    "Number of runs per experiment",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, float("inf")],
)

# 에러 및 상태 메트릭
mlflow_errors_total = Counter(
    "mlflow_errors_total",
    "Total MLflow errors by type",
    ["error_type", "operation", "experiment_name"],
)

# 시스템 정보 메트릭
mlflow_service_info = Info("mlflow_service_info", "MLflow service information")

# 성능 메트릭
mlflow_client_connections = Gauge(
    "mlflow_client_connections", "Number of active MLflow client connections"
)

mlflow_queue_size = Gauge("mlflow_queue_size", "Size of MLflow operation queue")


class MLflowTrackingService:
    """MLflow 추적 서비스 - 실험 관리 및 모델 버전 관리 (2025년 Prometheus 확장)"""

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

        # 서비스 정보 설정
        mlflow_service_info.info(
            {
                "version": "2.8.0",
                "tracking_uri": self.tracking_uri,
                "default_experiment": self.experiment_name,
                "service_start_time": datetime.now().isoformat(),
            }
        )

        # MLflow 2.8+ 비동기 로깅 활성화
        self._enable_async_logging()

        if client is None:
            self._initialize_client()

    def _enable_async_logging(self) -> None:
        """MLflow 2.8+ 비동기 로깅 활성화"""
        try:
            # MLflow 2.8+ 비동기 로깅 설정
            enable_async_logging(True)  # type: ignore
            logger.info("MLflow async logging enabled")
        except Exception as e:
            logger.warning(f"Failed to enable async logging: {e}")
            mlflow_errors_total.labels(
                error_type="async_logging_error",
                operation="initialization",
                experiment_name="system",
            ).inc()

    def _initialize_client(self) -> None:
        """MLflow 클라이언트 초기화"""
        try:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            mlflow_client_connections.inc()
            logger.info(f"MLflow client initialized with URI: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            mlflow_errors_total.labels(
                error_type="client_init_error", operation="initialization", experiment_name="system"
            ).inc()
            self.client = None

    def _ensure_client(self) -> bool:
        """클라이언트가 초기화되었는지 확인"""
        if self.client is None:
            self._initialize_client()
        return self.client is not None

    @contextmanager
    def _track_operation(self, operation: str, experiment_name: str = "unknown") -> Any:
        """Prometheus 메트릭 추적 컨텍스트"""
        start_time = time.time()
        try:
            yield
            mlflow_operations_total.labels(
                operation=operation, status="success", experiment_name=experiment_name
            ).inc()
        except Exception as e:
            mlflow_operations_total.labels(
                operation=operation, status="error", experiment_name=experiment_name
            ).inc()
            mlflow_errors_total.labels(
                error_type=type(e).__name__, operation=operation, experiment_name=experiment_name
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            mlflow_operation_duration.labels(operation=operation).observe(duration)
            logger.debug(f"Operation {operation} took {duration:.2f}s")

    def create_experiment(self, name: str, tags: dict[str, str] | None = None) -> str | None:
        """실험 생성"""
        with self._track_operation("create_experiment", name):
            if not self._ensure_client():
                mlflow_experiment_counter.labels(status="error").inc()
                return None

            assert self.client is not None
            try:
                experiment = self.client.get_experiment_by_name(name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(name, tags=tags or {})
                    if experiment_id is not None:
                        logger.info(f"Created experiment: {name} with ID: {experiment_id}")
                        mlflow_experiment_counter.labels(status="created").inc()
                        mlflow_experiments_active.inc()
                        return experiment_id
                    else:
                        mlflow_experiment_counter.labels(status="error").inc()
                        return None
                else:
                    logger.info(f"Experiment {name} already exists")
                    mlflow_experiment_counter.labels(status="existing").inc()
                    return str(experiment.experiment_id)
            except Exception as e:
                logger.error(f"Failed to create experiment {name}: {e}")
                mlflow_experiment_counter.labels(status="error").inc()
                return None

    def start_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """실행 시작"""
        exp_name = experiment_name or self.experiment_name
        with self._track_operation("start_run", exp_name):
            if not self._ensure_client():
                return None

            assert self.client is not None
            try:
                if experiment_name:
                    experiment = self.client.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = self.client.create_experiment(
                            experiment_name, tags=tags or {}
                        )
                        if experiment_id is None:
                            return None
                    else:
                        experiment_id = experiment.experiment_id
                else:
                    # 기본 실험 사용
                    experiment_id = "0"

                run = self.client.create_run(
                    experiment_id=experiment_id, run_name=run_name, tags=tags
                )
                logger.info(f"Started run: {run.info.run_id}")

                # 활성 run 수 업데이트
                self._update_active_runs_count()

                return str(run.info.run_id)
            except Exception as e:
                logger.error(f"Failed to start run: {e}")
                return None

    def _update_active_runs_count(self) -> None:
        """활성 run 수 업데이트"""
        try:
            if self.client:
                active_runs = self.client.search_runs(
                    experiment_ids=["0"], filter_string="status = 'RUNNING'"
                )
                mlflow_active_runs.set(len(active_runs))

                # 실험별 run 수 메트릭 업데이트
                experiments = self.client.search_experiments()
                for exp in experiments:
                    exp_runs = self.client.search_runs(experiment_ids=[exp.experiment_id])
                    mlflow_runs_per_experiment.observe(len(exp_runs))

        except Exception as e:
            logger.warning(f"Failed to update active runs count: {e}")

    def log_parameters(self, run_id: str, parameters: dict[str, Any]) -> bool:
        """파라미터 로깅"""
        with self._track_operation("log_parameters"):
            if not self._ensure_client():
                return False

            assert self.client is not None
            try:
                for key, value in parameters.items():
                    self.client.log_param(run_id, key, value)

                # 파라미터 수 메트릭 기록
                mlflow_parameters_total.labels(experiment_name="unknown").inc(len(parameters))

                logger.info(f"Logged {len(parameters)} parameters to run {run_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to log parameters: {e}")
                return False

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None = None) -> bool:
        """메트릭 로깅"""
        with self._track_operation("log_metrics"):
            if not self._ensure_client():
                return False

            assert self.client is not None
            try:
                for key, value in metrics.items():
                    self.client.log_metric(run_id, key, value, step=step)

                # 메트릭 수 메트릭 기록
                mlflow_metrics_total.labels(experiment_name="unknown").inc(len(metrics))

                logger.info(f"Logged {len(metrics)} metrics to run {run_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
                return False

    def log_model(
        self, run_id: str, model: Any, model_name: str, model_type: str = "sklearn"
    ) -> bool:
        """모델 로깅"""
        with self._track_operation("log_model", model_type):
            if not self._ensure_client():
                mlflow_model_logging_total.labels(
                    model_type=model_type, status="error", experiment_name="unknown"
                ).inc()
                return False

            try:
                if model_type == "sklearn":
                    # run_id를 사용하여 특정 실행에 모델 로깅
                    with mlflow.start_run(run_id=run_id):
                        sklearn.log_model(model, model_name)

                        # 모델 크기 추정
                        model_size = len(pickle.dumps(model))
                        mlflow_model_size_bytes.labels(model_type=model_type).observe(model_size)

                else:
                    logger.warning(f"Unsupported model type: {model_type}")
                    mlflow_model_logging_total.labels(
                        model_type=model_type, status="error", experiment_name="unknown"
                    ).inc()
                    return False

                logger.info(f"Logged model {model_name} to run {run_id}")
                mlflow_model_logging_total.labels(
                    model_type=model_type, status="success", experiment_name="unknown"
                ).inc()
                return True
            except Exception as e:
                logger.error(f"Failed to log model to run {run_id}: {e}")
                mlflow_model_logging_total.labels(
                    model_type=model_type, status="error", experiment_name="unknown"
                ).inc()
                return False

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> bool:
        """아티팩트 로깅"""
        with self._track_operation("log_artifact"):
            if not self._ensure_client():
                return False

            assert self.client is not None
            try:
                # 아티팩트 크기 측정
                import os

                artifact_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                artifact_type = (
                    os.path.splitext(local_path)[1][1:] if "." in local_path else "unknown"
                )

                self.client.log_artifact(run_id, local_path, artifact_path)

                # 아티팩트 메트릭 기록
                mlflow_artifacts_total.labels(
                    artifact_type=artifact_type, status="success", experiment_name="unknown"
                ).inc()
                mlflow_artifact_size_bytes.labels(artifact_type=artifact_type).observe(
                    artifact_size
                )

                logger.info(f"Logged artifact {local_path} to run {run_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to log artifact to run {run_id}: {e}")
                return False

    def end_run(self, run_id: str, status: str = "FINISHED") -> bool:
        """실행 종료"""
        with self._track_operation("end_run"):
            if not self._ensure_client():
                return False

            assert self.client is not None
            try:
                # Run 시작 시간 가져오기
                run_info = self.client.get_run(run_id).info
                run_start_time = run_info.start_time / 1000  # 밀리초를 초로 변환

                self.client.set_terminated(run_id, status)
                logger.info(f"Ended run {run_id} with status {status}")

                # 실행 시간 기록
                duration = time.time() - run_start_time
                mlflow_run_duration.labels(experiment_name=run_info.experiment_id).observe(duration)

                # 활성 run 수 업데이트
                self._update_active_runs_count()

                return True
            except Exception as e:
                logger.error(f"Failed to end run {run_id}: {e}")
                return False

    def get_run_history(self, experiment_name: str | None = None) -> dict[str, Any]:
        """실행 기록 조회"""
        exp_name = experiment_name or "all"
        with self._track_operation("get_run_history", exp_name):
            if not self._ensure_client():
                return {"runs": []}

            assert self.client is not None
            try:
                runs_list: list[Any] = []

                if experiment_name:
                    experiment = self.client.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = self.client.search_runs(
                            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
                        )
                        runs_list = runs
                    else:
                        runs_list = []
                else:
                    runs = self.client.search_runs(
                        experiment_ids=["0"], order_by=["start_time DESC"]
                    )
                    runs_list = runs

                return {
                    "runs": [
                        {
                            "run_id": run.info.run_id,
                            "experiment_id": run.info.experiment_id,
                            "status": run.info.status,
                            "start_time": run.info.start_time,
                            "end_time": run.info.end_time,
                            "metrics": run.data.metrics,
                            "params": run.data.params,
                        }
                        for run in runs_list[:50]
                    ]
                }

            except Exception as e:
                logger.error(f"Failed to get run history: {e}")
                return {"runs": []}

    def register_model(
        self,
        model_name: str,
        run_id: str,
        model_path: str = "model",
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """모델 등록"""
        with self._track_operation("register_model", model_name):
            if not self._ensure_client():
                return {"status": "error", "message": "MLflow client not available"}

            assert self.client is not None
            try:
                model_uri = f"runs:/{run_id}/{model_path}"
                model_version = self.client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run_id,
                    description=description,
                    tags=tags,
                )

                return {
                    "status": "success",
                    "model_name": model_name,
                    "model_version": model_version.version,
                    "run_id": run_id,
                }
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                return {"status": "error", "error": str(e)}

    def get_experiment_metrics(self, experiment_name: str) -> dict[str, Any]:
        """실험 메트릭 집계"""
        with self._track_operation("get_experiment_metrics", experiment_name):
            if not self._ensure_client():
                return {"error": "MLflow client not initialized"}

            assert self.client is not None
            try:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if not experiment:
                    return {"error": f"Experiment {experiment_name} not found"}

                runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])

                # 메트릭 집계
                all_metrics: dict[str, list[float]] = {}
                for run in runs:
                    for metric_name, metric_value in run.data.metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)

                # 통계 계산
                metrics_summary = {}
                for metric_name, values in all_metrics.items():
                    if values:
                        metrics_summary[metric_name] = {
                            "count": len(values),
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "latest": values[-1],  # 가장 최신 값
                        }

                return {
                    "experiment": experiment_name,
                    "total_runs": len(runs),
                    "metrics_summary": metrics_summary,
                }

            except Exception as e:
                logger.error(f"Failed to get experiment metrics: {e}")
                return {"error": str(e)}

    def cleanup_failed_runs(self, experiment_name: str | None = None) -> dict[str, Any]:
        """실패한 실행 정리"""
        with self._track_operation("cleanup_failed_runs", experiment_name or "all"):
            if not self._ensure_client():
                return {"error": "MLflow client not initialized"}

            assert self.client is not None
            try:
                if experiment_name:
                    experiment = self.client.get_experiment_by_name(experiment_name)
                    if not experiment:
                        return {"error": f"Experiment {experiment_name} not found"}

                    failed_runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id], filter_string="status = 'FAILED'"
                    )
                else:
                    failed_runs = self.client.search_runs(
                        experiment_ids=["0"], filter_string="status = 'FAILED'"
                    )

                cleaned_count = 0
                for run in failed_runs:
                    try:
                        self.client.delete_run(run.info.run_id)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete run {run.info.run_id}: {e}")

                logger.info(f"Cleaned up {cleaned_count} failed runs")

                return {
                    "status": "success",
                    "cleaned_count": cleaned_count,
                    "experiment": experiment_name or "all",
                }

            except Exception as e:
                logger.error(f"Failed to cleanup runs: {e}")
                return {"error": str(e)}

    def get_model_performance_trend(
        self, model_name: str, metric_name: str = "accuracy"
    ) -> dict[str, Any]:
        """모델 성능 추세 분석"""
        with self._track_operation("get_model_trend", model_name):
            if not self._ensure_client():
                return {"error": "MLflow client not initialized"}

            assert self.client is not None
            try:
                # 모델 관련 실행 검색
                runs = self.client.search_runs(
                    experiment_ids=["0"], filter_string=f"tags.mlflow.runName LIKE '%{model_name}%'"
                )

                # 시간순 정렬 및 메트릭 추출
                trend_data = []
                for run in sorted(runs, key=lambda x: x.info.start_time or 0):
                    if metric_name in run.data.metrics:
                        trend_data.append(
                            {
                                "run_id": run.info.run_id,
                                "timestamp": run.info.start_time,
                                "value": run.data.metrics[metric_name],
                                "status": run.info.status,
                            }
                        )

                return {
                    "model": model_name,
                    "metric": metric_name,
                    "trend": trend_data,
                    "total_runs": len(trend_data),
                }

            except Exception as e:
                logger.error(f"Failed to get model trend: {e}")
                return {"error": str(e)}

    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.client:
                # 활성 run 정리
                active_runs = self.client.search_runs(
                    experiment_ids=["0"], filter_string="status = 'RUNNING'"
                )
                for run in active_runs:
                    try:
                        self.client.set_terminated(run.info.run_id, "FAILED")
                    except Exception as e:
                        logger.warning(f"Failed to terminate run {run.info.run_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup MLflow resources: {e}")

    def __del__(self) -> None:
        """리소스 정리"""
        self.cleanup()


def get_mlflow_tracking_service() -> MLflowTrackingService:
    """MLflow 추적 서비스 인스턴스 생성 (lazy initialization)"""
    return MLflowTrackingService()
