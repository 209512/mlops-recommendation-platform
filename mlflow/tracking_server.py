import os

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import mlflow


class MLflowTrackingServer:
    """MLflow 추천 서버 - 실험 추적 및 모델 관리"""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        # None이 아님을 보장하여 MyPy 오류 해결
        if self.tracking_uri is None:
            self.tracking_uri = "http://localhost:5000"

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    def create_experiment(self, name: str, tags: dict[str, str] | None = None) -> str:
        """실험 생성"""
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            return str(experiment.experiment_id)

        experiment_id = self.client.create_experiment(
            name=name, tags=tags or {"framework": "FastAPI", "model_type": "ALS"}
        )
        return str(experiment_id)

    def log_model_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """모델 메트릭 기록"""
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)

    def register_model(self, run_id: str, model_name: str, model_path: str = "als_model") -> str:
        """모델 등록"""
        model_uri = f"runs:/{run_id}/{model_path}"
        model_version = self.client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        return str(model_version.version)


def start_server(
    host: str = "0.0.0.0", port: int = 5000, default_artifact_root: str = "./mlruns"
) -> None:
    """MLflow 서버 시작"""
    os.system(
        f"mlflow server --host {host} --port {port} --default-artifact-root {default_artifact_root}"
    )


if __name__ == "__main__":
    start_server()
