import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import psutil
from prometheus_client import Counter, Gauge, Histogram

import mlflow
from app.core.config import settings
from app.infrastructure.redis import get_redis_client
from app.services.mlflow.tracking import get_mlflow_tracking_service

logger = logging.getLogger(__name__)


class MLOpsMonitoring:
    """MLOps 시스템 모니터링 - Prometheus 메트릭 수집"""

    def __init__(self) -> None:
        # 추천 시스템 메트릭
        self.recommendation_counter = Counter(
            "recommendations_total", "Total recommendations generated", ["user_id", "model_version"]
        )

        self.recommendation_latency = Histogram(
            "recommendation_latency_seconds",
            "Time spent generating recommendations",
            ["model_version"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # 모델 학습 메트릭
        self.training_counter = Counter(
            "model_training_total", "Total model training runs", ["model_type", "status"]
        )

        self.training_duration = Histogram(
            "model_training_duration_seconds",
            "Model training duration",
            ["model_type"],
            buckets=[60, 300, 900, 1800, 3600, 7200],  # 1분~2시간
        )

        self.model_accuracy = Gauge("model_accuracy", "Current model accuracy", ["model_version"])

        # 시스템 메트릭
        self.system_memory_usage = Gauge(
            "system_memory_usage_percent", "System memory usage percentage"
        )

        self.system_cpu_usage = Gauge("system_cpu_usage_percent", "System CPU usage percentage")

        self.redis_connection_pool = Gauge(
            "redis_connection_pool_active", "Active Redis connections"
        )

        # MLflow 메트릭
        self.mlflow_experiment_count = Gauge("mlflow_experiments_total", "Total MLflow experiments")

        self.mlflow_run_count = Counter(
            "mlflow_runs_total", "Total MLflow runs", ["experiment_id", "status"]
        )

        self.redis_client = get_redis_client()

    def track_recommendation(
        self, model_version: str = "latest"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """추천 생성 추적 데코레이터"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # 성공 메트릭 기록
                    duration = time.time() - start_time
                    self.recommendation_latency.labels(model_version=model_version).observe(
                        duration
                    )

                    # 사용자 ID가 있다면 라벨에 추가
                    if len(args) > 0 and isinstance(args[0], int):
                        user_id = str(args[0])
                        self.recommendation_counter.labels(
                            user_id=user_id, model_version=model_version
                        ).inc()
                    else:
                        self.recommendation_counter.labels(
                            user_id="unknown", model_version=model_version
                        ).inc()

                    return result

                except Exception as e:
                    logger.error(f"Recommendation failed: {e}")
                    # 실패 메트릭 기록
                    self.recommendation_counter.labels(
                        user_id="error", model_version=model_version
                    ).inc()
                    raise

            return wrapper

        return decorator

    def track_training(
        self, model_type: str = "als"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """모델 학습 추적 데코레이터"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # 성공 메트릭 기록
                    duration = time.time() - start_time
                    self.training_duration.labels(model_type=model_type).observe(duration)
                    self.training_counter.labels(model_type=model_type, status="success").inc()

                    # 결과에서 정확도 추출 (있는 경우)
                    if isinstance(result, dict) and "accuracy" in result:
                        self.model_accuracy.labels(model_version="latest").set(result["accuracy"])

                    return result

                except Exception as e:
                    logger.error(f"Model training failed: {e}")
                    # 실패 메트릭 기록
                    self.training_counter.labels(model_type=model_type, status="failed").inc()
                    raise

            return wrapper

        return decorator

    def update_system_metrics(self) -> None:
        """시스템 메트릭 업데이트"""
        try:
            # 메모리 사용량
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)

            # CPU 사용량
            cpu = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu)

            # Redis 연결 풀
            if hasattr(self.redis_client, "connection_pool"):
                pool = self.redis_client.connection_pool
                if hasattr(pool, "created_connections"):
                    self.redis_connection_pool.set(pool.created_connections)

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def update_mlflow_metrics(self) -> None:
        """MLflow 메트릭 업데이트"""
        try:
            client = mlflow.tracking.MlflowClient()

            # 실험 수
            experiments = client.search_experiments()
            self.mlflow_experiment_count.set(len(experiments))

            # 최근 실행 수
            for experiment in experiments:
                runs = client.search_runs(experiment.experiment_id)
                for run in runs:
                    status = run.info.status.lower()
                    self.mlflow_run_count.labels(
                        experiment_id=experiment.experiment_id, status=status
                    ).inc()

        except Exception as e:
            logger.error(f"Failed to update MLflow metrics: {e}")

    def check_health(self) -> bool:
        """시스템 건강 상태 확인"""
        try:
            # Redis 연결 확인
            self.redis_client.ping()

            # MLflow 연결 확인
            mlflow.tracking.MlflowClient().search_experiments()

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_system_info(self) -> dict[str, Any]:
        """시스템 정보 조회"""
        try:
            return {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "disk_usage": psutil.disk_usage("/").percent,
                "redis_connected": self.redis_client.ping(),
                "mlflow_tracking_uri": settings.mlflow_tracking_uri,  # 수정: 소문자로 변경
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """활성 알림 조회"""
        alerts: list[dict[str, Any]] = []

        try:
            # 메모리 사용량 경고
            memory = psutil.virtual_memory().percent
            if memory > 90:
                alerts.append(
                    {
                        "type": "memory",
                        "severity": "critical",
                        "message": f"Memory usage is {memory:.1f}%",
                        "threshold": 90,
                    }
                )
            elif memory > 80:
                alerts.append(
                    {
                        "type": "memory",
                        "severity": "warning",
                        "message": f"Memory usage is {memory:.1f}%",
                        "threshold": 80,
                    }
                )

            # CPU 사용량 경고
            cpu = psutil.cpu_percent(interval=1)
            if cpu > 90:
                alerts.append(
                    {
                        "type": "cpu",
                        "severity": "critical",
                        "message": f"CPU usage is {cpu:.1f}%",
                        "threshold": 90,
                    }
                )
            elif cpu > 80:
                alerts.append(
                    {
                        "type": "cpu",
                        "severity": "warning",
                        "message": f"CPU usage is {cpu:.1f}%",
                        "threshold": 80,
                    }
                )

                # 디스크 사용량 경고
            disk = psutil.disk_usage("/").percent
            if disk > 90:
                alerts.append(
                    {
                        "type": "disk",
                        "severity": "critical",
                        "message": f"Disk usage is {disk:.1f}%",
                        "threshold": 90,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            alerts.append(
                {
                    "type": "system",
                    "severity": "error",
                    "message": f"Failed to check alerts: {str(e)}",
                }
            )

        return alerts

    def log_model_metrics(self, metrics: dict[str, float], model_version: str = "latest") -> None:
        try:
            # MLflowTrackingService 사용
            mlflow_service = get_mlflow_tracking_service()
            if mlflow_service and mlflow_service.client:
                # experiment_id 동적 조회
                experiment = mlflow_service.client.get_experiment_by_name(
                    mlflow_service.experiment_name
                )
                experiment_id = experiment.experiment_id if experiment else None

                # 임시 실행 생성
                run = mlflow_service.client.create_run(experiment_id=experiment_id or "0")

                # 메트릭 기록
                for metric_name, value in metrics.items():
                    mlflow_service.client.log_metric(run.info.run_id, metric_name, value)

                # 태그 설정
                mlflow_service.client.set_tag(run.info.run_id, "model_version", model_version)
        except Exception as e:
            logger.error(f"Failed to log model metrics: {e}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """메트릭 요약 조회"""
        try:
            return {
                "recommendations_total": self.recommendation_counter._value.get(),
                "avg_recommendation_latency": 0.0,  # 수정: Histogram에서 평균값 직접 조회 불가
                "model_accuracy": self.model_accuracy._value.get(),
                "system_memory_usage": self.system_memory_usage._value.get(),
                "system_cpu_usage": self.system_cpu_usage._value.get(),
                "active_alerts": len(self.get_active_alerts()),
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}

    def cleanup(self) -> None:
        """리소스 정리"""
        logger.info("Cleaning up monitoring resources")


def get_monitoring_service() -> MLOpsMonitoring:
    """MLOps 모니터링 서비스 인스턴스 생성"""
    return MLOpsMonitoring()
